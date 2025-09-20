import datetime
import os
import sys
import time
import warnings
import registry
from pathlib import Path
import torch.distributed as dist

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
warnings.filterwarnings("ignore", category=UserWarning)

from utils.imagenet_utils import presets, transforms, utils, sampler
import torch
import torch.utils.data
import torchvision
from torch import nn
from torch.utils.data.dataloader import default_collate
from torch.func import functional_call

import torch_pruning as tp
from omegaconf import DictConfig, OmegaConf
import hydra

import matplotlib.pyplot as plt

from utils.convert import state_dict_to_graph, graph_to_state_dict, state_dict_to_model, graph_to_model
from utils.pruner import get_pruner
from utils.pruning import progressive_pruning


def prune_to_target_flops(pruner, model, target_flops, example_inputs, cfg):
    model.eval()
    ori_ops, _ = tp.utils.count_ops_and_params(model, example_inputs=example_inputs)
    pruned_ops = ori_ops
    while pruned_ops / 1e9 > target_flops:
        pruner.step()
        if "vit" in cfg.model:
            model.hidden_dim = model.conv_proj.out_channels
        pruned_ops, _ = tp.utils.count_ops_and_params(model, example_inputs=example_inputs)
    return pruned_ops


def train_one_epoch(
    model, criterion, optimizer, data_loader, device, epoch, cfg, scaler=None, pruner=None, recover=None
):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value}"))
    metric_logger.add_meter("img/s", utils.SmoothedValue(window_size=10, fmt="{value}"))

    header = f"Epoch: [{epoch}]"
    for i, (image, target) in enumerate(metric_logger.log_every(data_loader, cfg.print_freq, header)):
        start_time = time.time()
        image, target = image.to(device), target.to(device)
        # with torch.cuda.amp.autocast(enabled=scaler is not None):
        with torch.amp.autocast("cuda", enabled=scaler is not None):
            output = model(image)
            loss = criterion(output, target)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            if pruner:
                scaler.unscale_(optimizer)
                pruner.regularize(model)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if pruner is not None:
                pruner.regularize(model)
            if recover:
                recover(model.module)
            if cfg.clip_grad_norm is not None:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_grad_norm)
            optimizer.step()

        acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
        batch_size = image.shape[0]
        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
        metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
        metric_logger.meters["img/s"].update(batch_size / (time.time() - start_time))
    if pruner is not None and isinstance(pruner, tp.pruner.GrowingRegPruner):
        pruner.update_reg()


def meta_train_one_epoch(
    data_model_num, metanetwork, criterion, optimizer, big_data_loader, device, 
    epoch, cfg_meta_train, cfg
):
    model_name = cfg.model
    metanetwork.train()
    index = cfg.rank % data_model_num
    ckpt = torch.load(os.path.join('save', f'{cfg.name}', 'meta_train', 'data_model', f'{index}.pth'), weights_only=False, map_location=device)
    origin_state_dict = ckpt['model']

    node_index, node_features, edge_index, edge_features_list = state_dict_to_graph(model_name, origin_state_dict)
    model = state_dict_to_model(model_name, origin_state_dict, device)
    pruner = get_pruner(model, torch.ones((1, 3, 224, 224)).to(device), cfg_meta_train.pruner_reg, "IMAGENET", cfg.method, special_type=cfg.model.split('_')[0])

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value}"))
    metric_logger.add_meter("img/s", utils.SmoothedValue(window_size=10, fmt="{value}"))

    header = f"Epoch: [{epoch}]"
    state_dict = None
    for i, (image, target) in enumerate(metric_logger.log_every(big_data_loader, cfg_meta_train.print_freq, header)):
        def _one_step(i, image, target):
            if i >= (len(big_data_loader) // cfg.meta_train.optimize_every_iter) * cfg.meta_train.optimize_every_iter:
                return
            start_time = time.time()
            nonlocal state_dict
            if i % cfg_meta_train.optimize_every_iter == 0:
                node_pred, edge_pred = metanetwork.forward(node_features, edge_index, edge_features_list)
                state_dict = graph_to_state_dict(model_name, origin_state_dict, node_index, node_pred, edge_index, edge_pred, device)
                optimizer.zero_grad()
            image, target = image.to(device), target.to(device)
            output = functional_call(model, state_dict, image)
            loss = criterion(output, target)
            loss.backward(retain_graph=True)
            if (i + 1) % cfg_meta_train.optimize_every_iter == 0:
                model.load_state_dict(state_dict)
                for param in model.parameters():
                    if param.requires_grad:
                        param.grad = torch.zeros_like(param)
                pruner.regularize(model, alpha=cfg_meta_train.alpha, bias=cfg_meta_train.bias)
                big_tensor = []
                big_gradient = []
                for name, param in model.named_parameters():
                    if param.requires_grad:
                        if not torch.isnan(param.grad).any():
                            big_tensor.append(state_dict[name].flatten())
                            big_gradient.append(param.grad.flatten())
                big_tensor = torch.cat(big_tensor, dim=0)
                big_gradient = torch.cat(big_gradient, dim=0)
                big_tensor.backward(gradient=big_gradient)
            if (i + 1) % cfg_meta_train.optimize_every_iter == 0:
                # if cfg_meta_train.clip_grad_norm != 0:
                #     nn.utils.clip_grad_norm_(metanetwork.parameters(), cfg_meta_train.clip_grad_norm)
                # max_grad = max(p.grad.abs().max() for p in metanetwork.parameters() if p.grad is not None)
                # print(f"Max gradient: {max_grad.item()}")
                # optimizer.step()

                if cfg_meta_train.clip_grad_norm != 0:
                    # Compute the total gradient norm BEFORE clipping
                    total_norm_before = torch.norm(
                        torch.stack([torch.norm(p.grad.detach(), 2) for p in metanetwork.parameters() if p.grad is not None]), 
                        2
                    )
                    
                    # Clip gradients
                    nn.utils.clip_grad_norm_(metanetwork.parameters(), cfg_meta_train.clip_grad_norm)
                    
                    # Compute the total gradient norm AFTER clipping
                    total_norm_after = torch.norm(
                        torch.stack([torch.norm(p.grad.detach(), 2) for p in metanetwork.parameters() if p.grad is not None]), 
                        2
                    )
                    
                    # Check if gradients were clipped
                    if total_norm_after < total_norm_before:
                        print(f"Gradients were CLIPPED (before: {total_norm_before:.4f}, after: {total_norm_after:.4f})")
                    else:
                        print(f"Gradients were NOT clipped (norm = {total_norm_before:.4f})")
                
                optimizer.step()

            acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
            batch_size = image.shape[0]
            metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])
            metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
            metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
            metric_logger.meters["img/s"].update(batch_size / (time.time() - start_time))
        _one_step(i, image, target)
    del pruner

    return metric_logger.loss.global_avg, metric_logger.acc1.global_avg, metric_logger.acc5.global_avg


def evaluate(model, criterion, data_loader, device, print_freq=200, log_suffix=""):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = f"Test: {log_suffix}"

    num_processed_samples = 0
    with torch.no_grad():
        for image, target in metric_logger.log_every(data_loader, print_freq, header):
            image = image.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            output = model(image)
            loss = criterion(output, target)

            acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
            batch_size = image.shape[0]
            metric_logger.update(loss=loss.item())
            metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
            metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
            num_processed_samples += batch_size
    num_processed_samples = utils.reduce_across_processes(num_processed_samples)
    if (
        hasattr(data_loader.dataset, "__len__")
        and len(data_loader.dataset) != num_processed_samples
        and torch.distributed.get_rank() == 0
    ):
        warnings.warn(
            f"Dataset length mismatch: {len(data_loader.dataset)} vs {num_processed_samples}"
        )

    metric_logger.synchronize_between_processes()
    print(f"{header} Acc@1 {metric_logger.acc1.global_avg:.3f} Acc@5 {metric_logger.acc5.global_avg:.3f}")
    return metric_logger.acc1.global_avg, metric_logger.acc5.global_avg


def _get_cache_path(filepath):
    import hashlib

    h = hashlib.sha1(filepath.encode()).hexdigest()
    cache_path = os.path.join("~", ".torch", "vision", "datasets", "imagefolder", h[:10] + ".pt")
    cache_path = os.path.expanduser(cache_path)
    return cache_path


def load_data(traindir, valdir, cfg):
    resize_size, crop_size = (342, 299) if cfg.model == "inception_v3" else (256, 224)

    print("Loading training data...")
    cache_path = _get_cache_path(traindir)
    if cfg.cache_dataset and os.path.exists(cache_path):
        dataset, _ = torch.load(cache_path, weights_only=False)
    else:
        auto_augment_policy = getattr(cfg, "auto_augment", None)
        random_erase_prob = getattr(cfg, "random_erase", 0.0)
        dataset = torchvision.datasets.ImageFolder(
            traindir,
            presets.ClassificationPresetTrain(
                crop_size=crop_size,
                auto_augment_policy=auto_augment_policy,
                random_erase_prob=random_erase_prob,
            ),
        )
        if cfg.cache_dataset:
            utils.mkdir(os.path.dirname(cache_path))
            utils.save_on_master((dataset, traindir), cache_path)

    print("Loading validation data...")
    cache_path = _get_cache_path(valdir)
    if cfg.cache_dataset and os.path.exists(cache_path):
        dataset_test, _ = torch.load(cache_path, weights_only=False)
    else:
        dataset_test = torchvision.datasets.ImageFolder(
            valdir, presets.ClassificationPresetEval(crop_size=crop_size, resize_size=resize_size)
        )
        if cfg.cache_dataset:
            utils.mkdir(os.path.dirname(cache_path))
            utils.save_on_master((dataset_test, valdir), cache_path)

    print("Creating data loaders...")
    if cfg.distributed:######################################################################################################
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test)
    else:
        train_sampler = torch.utils.data.RandomSampler(dataset)
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    return dataset, dataset_test, train_sampler, test_sampler


def train(
    model, 
    epochs, 
    lr, lr_warmup_epochs, 
    train_sampler, data_loader, data_loader_test, 
    device, cfg, pruner=None, state_dict_only=True, save_every_epoch=False):

    model.to(device)
    if cfg.distributed and cfg.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    if cfg.label_smoothing>0:
        criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    else:
        criterion = nn.CrossEntropyLoss()

    weight_decay = cfg.weight_decay if pruner is None else 0
    bias_weight_decay = cfg.bias_weight_decay if pruner is None else 0
    norm_weight_decay = cfg.norm_weight_decay if pruner is None else 0

    custom_keys_weight_decay = []
    if bias_weight_decay is not None:
        custom_keys_weight_decay.append(("bias", bias_weight_decay))
    if cfg.transformer_embedding_decay is not None:
        for key in ["class_token", "position_embedding", "relative_position_bias_table"]:
            custom_keys_weight_decay.append((key, cfg.transformer_embedding_decay))
    parameters = utils.set_weight_decay(
        model,
        weight_decay,
        norm_weight_decay=norm_weight_decay,
        custom_keys_weight_decay=custom_keys_weight_decay if len(custom_keys_weight_decay) > 0 else None,
    )

    opt_name = cfg.opt.lower()
    if opt_name.startswith("sgd"):
        optimizer = torch.optim.SGD(
            parameters,
            lr=lr,
            momentum=cfg.momentum,
            weight_decay=weight_decay,
            nesterov="nesterov" in opt_name,
        )
    elif opt_name == "rmsprop":
        optimizer = torch.optim.RMSprop(
            parameters, lr=lr, momentum=cfg.momentum, weight_decay=weight_decay, eps=0.0316, alpha=0.9
        )
    elif opt_name == "adamw":
        optimizer = torch.optim.AdamW(parameters, lr=lr, weight_decay=weight_decay)
    else:
        raise RuntimeError(f"Invalid optimizer {cfg.opt}. Only SGD, RMSprop and AdamW are supported.")

    # scaler = torch.cuda.amp.GradScaler() if args.amp else None
    scaler = torch.amp.GradScaler('cuda') if cfg.amp else None

    cfg.lr_scheduler = cfg.lr_scheduler.lower()
    if cfg.lr_scheduler == "steplr":
        milestones = [int(ms) for ms in cfg.lr_decay_milestones.split(",")]
        main_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=milestones, gamma=cfg.lr_gamma
        )
    elif cfg.lr_scheduler == "cosineannealinglr":
        main_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs - lr_warmup_epochs, eta_min=cfg.lr_min
        )
    elif cfg.lr_scheduler == "exponentiallr":
        main_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=cfg.lr_gamma)
    else:
        raise RuntimeError(
            f"Invalid lr scheduler '{cfg.lr_scheduler}'. Only StepLR, CosineAnnealingLR and ExponentialLR "
            "are supported."
        )

    if lr_warmup_epochs > 0:
        if cfg.lr_warmup_method == "linear":
            warmup_lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=cfg.lr_warmup_decay, total_iters=lr_warmup_epochs
            )
        elif cfg.lr_warmup_method == "constant":
            warmup_lr_scheduler = torch.optim.lr_scheduler.ConstantLR(
                optimizer, factor=cfg.lr_warmup_decay, total_iters=lr_warmup_epochs
            )
        else:
            raise RuntimeError(
                f"Invalid warmup lr method '{cfg.lr_warmup_method}'. Only linear and constant are supported."
            )
        lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_lr_scheduler, main_lr_scheduler], milestones=[lr_warmup_epochs]
        )
    else:
        lr_scheduler = main_lr_scheduler

    model_without_ddp = model
    if cfg.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[cfg.gpu])
        model_without_ddp = model.module

    if cfg.resume:
        print('resume from {}'.format(cfg.resume))
        checkpoint = torch.load(cfg.resume, map_location="cpu", weights_only=False)
        model_without_ddp.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        # lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        cfg.lr_scheduler = cfg.lr_scheduler.lower()
        if cfg.lr_scheduler == "steplr":
            milestones = [int(ms) for ms in cfg.lr_decay_milestones.split(",")]
            main_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=milestones, gamma=cfg.lr_gamma
            )
        elif cfg.lr_scheduler == "cosineannealinglr":
            main_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=epochs - lr_warmup_epochs, eta_min=cfg.lr_min
            )
        elif cfg.lr_scheduler == "exponentiallr":
            main_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=cfg.lr_gamma)
        else:
            raise RuntimeError(
                f"Invalid lr scheduler '{cfg.lr_scheduler}'. Only StepLR, CosineAnnealingLR and ExponentialLR "
                "are supported."
            )

        if lr_warmup_epochs > 0:
            if cfg.lr_warmup_method == "linear":
                warmup_lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                    optimizer, start_factor=cfg.lr_warmup_decay, total_iters=lr_warmup_epochs
                )
            elif cfg.lr_warmup_method == "constant":
                warmup_lr_scheduler = torch.optim.lr_scheduler.ConstantLR(
                    optimizer, factor=cfg.lr_warmup_decay, total_iters=lr_warmup_epochs
                )
            else:
                raise RuntimeError(
                    f"Invalid warmup lr method '{cfg.lr_warmup_method}'. Only linear and constant are supported."
                )
            lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer, schedulers=[warmup_lr_scheduler, main_lr_scheduler], milestones=[lr_warmup_epochs]
            )
        else:
            lr_scheduler = main_lr_scheduler

        cfg.start_epoch = checkpoint["epoch"] + 1
        for i in range(cfg.start_epoch):
            lr_scheduler.step()
        if scaler:
            scaler.load_state_dict(checkpoint["scaler"])
    
    start_time = time.time()
    best_acc = 0
    prefix = '' if pruner is None else 'regularized_{:e}_'.format(cfg.reg)
    for epoch in range(cfg.start_epoch, epochs):
        if cfg.distributed:
            train_sampler.set_epoch(epoch)
        train_one_epoch(model, criterion, optimizer, data_loader, device, epoch, cfg, scaler, pruner)
        lr_scheduler.step()
        acc, _ = evaluate(model, criterion, data_loader_test, device=device)
        if cfg.output_dir:
            checkpoint = {
                "model": model_without_ddp.state_dict() if state_dict_only else model_without_ddp,
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "epoch": epoch,
                "args": cfg,
            }
            if scaler:
                checkpoint["scaler"] = scaler.state_dict()
            if acc>best_acc:
                best_acc=acc
                utils.save_on_master(checkpoint, os.path.join(cfg.output_dir, prefix+"best.pth"))
            if save_every_epoch:
                utils.save_on_master(checkpoint, os.path.join(cfg.output_dir, prefix+"epoch_{}.pth".format(epoch)))
            utils.save_on_master(checkpoint, os.path.join(cfg.output_dir, prefix+"latest.pth"))
        print("Epoch {}/{}, Current Best Acc = {:.6f}".format(epoch, epochs, best_acc))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Training time {total_time_str}")
    return model_without_ddp


def meta_train(
    data_model_num,
    metanetwork,
    epochs, 
    lr, lr_warmup_epochs, 
    train_sampler, big_data_loader, 
    device, cfg_meta_train, cfg):


    if cfg_meta_train.label_smoothing>0:
        criterion = nn.CrossEntropyLoss(label_smoothing=cfg_meta_train.label_smoothing)
    else:
        criterion = nn.CrossEntropyLoss()

    weight_decay = cfg_meta_train.weight_decay 

    opt_name = cfg_meta_train.opt.lower()
    if opt_name.startswith("sgd"):
        optimizer = torch.optim.SGD(
            metanetwork.parameters(),
            lr=lr,
            momentum=cfg_meta_train.momentum,
            weight_decay=weight_decay,
            nesterov="nesterov" in opt_name,
        )
    elif opt_name == "rmsprop":
        optimizer = torch.optim.RMSprop(
            metanetwork.parameters(), lr=lr, momentum=cfg_meta_train.momentum, weight_decay=weight_decay, eps=0.0316, alpha=0.9
        )
    elif opt_name == "adamw":
        optimizer = torch.optim.AdamW(metanetwork.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise RuntimeError(f"Invalid optimizer {cfg_meta_train.opt}. Only SGD, RMSprop and AdamW are supported.")
    

    cfg_meta_train.lr_scheduler = cfg_meta_train.lr_scheduler.lower()
    if cfg_meta_train.lr_scheduler == "steplr":
        milestones = [int(ms) for ms in cfg_meta_train.lr_decay_milestones.split(",")]
        main_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=milestones, gamma=cfg_meta_train.lr_gamma
        )
    elif cfg_meta_train.lr_scheduler == "cosineannealinglr":
        main_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs - lr_warmup_epochs, eta_min=cfg_meta_train.lr_min
        )
    elif cfg_meta_train.lr_scheduler == "exponentiallr":
        main_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=cfg_meta_train.lr_gamma)
    else:
        raise RuntimeError(
            f"Invalid lr scheduler '{cfg_meta_train.lr_scheduler}'. Only StepLR, CosineAnnealingLR and ExponentialLR "
            "are supported."
        )

    if lr_warmup_epochs > 0:
        if cfg_meta_train.lr_warmup_method == "linear":
            warmup_lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=cfg_meta_train.lr_warmup_decay, total_iters=lr_warmup_epochs
            )
        elif cfg_meta_train.lr_warmup_method == "constant":
            warmup_lr_scheduler = torch.optim.lr_scheduler.ConstantLR(
                optimizer, factor=cfg_meta_train.lr_warmup_decay, total_iters=lr_warmup_epochs
            )
        else:
            raise RuntimeError(
                f"Invalid warmup lr method '{cfg_meta_train.lr_warmup_method}'. Only linear and constant are supported."
            )
        lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_lr_scheduler, main_lr_scheduler], milestones=[lr_warmup_epochs]
        )
    else:
        lr_scheduler = main_lr_scheduler

    model_without_ddp = metanetwork
    if cfg.distributed:
        metanetwork = torch.nn.parallel.DistributedDataParallel(metanetwork, device_ids=[cfg.gpu], static_graph=True, find_unused_parameters=True)
        model_without_ddp = metanetwork.module

    if cfg.resume:
        print('resume from {}'.format(cfg.resume))
        checkpoint = torch.load(cfg.resume, map_location="cpu", weights_only=False)
        model_without_ddp.load_state_dict(checkpoint["model"].state_dict())
        optimizer.load_state_dict(checkpoint["optimizer"])
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        # lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        milestones = [int(ms) for ms in cfg_meta_train.lr_decay_milestones.split(",")]
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=milestones, gamma=cfg_meta_train.lr_gamma
        )
        cfg.start_epoch = checkpoint["epoch"] + 1
        for i in range(cfg.start_epoch):
            lr_scheduler.step()
    
    start_time = time.time()
    for epoch in range(cfg.start_epoch, epochs):
        def _one_epoch():
            if cfg.distributed:
                train_sampler.set_epoch(epoch)
            train_loss, acc1, acc5 = meta_train_one_epoch(data_model_num, metanetwork, criterion, optimizer, big_data_loader, device,
                                                        epoch, cfg_meta_train, cfg)
            lr_scheduler.step()
            if cfg.output_dir:
                checkpoint = {
                    "model": model_without_ddp,
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "epoch": epoch,
                    "args": cfg,
                }
                utils.save_on_master(checkpoint, os.path.join('save', f'{cfg.name}', 'meta_train', 'metanetwork', f"epoch_{epoch}.pth"))

            print("[Meta Training] Epoch: {}/{}, Train loss: {}, Acc1: {}, Acc5: {}".format(epoch, epochs, train_loss, acc1, acc5))
        _one_epoch()
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Training time {total_time_str}")
    return model_without_ddp




def visualize_acc_speed_up_curve(
        models,
        labels,
        test_loader,
        base_speed_up,
        cfg,
        max_speed_up = 5.0,
        marker = 'o',
        save_dir ='tmp/',
        name = 'tmp.png',
        ylim = (0.0, 1.0),
        font_scale = 1.5,
):
    def get_acc_speed_up_list(
        model,
        test_loader,
        base_speed_up,
        max_speed_up = 5.0,
    ):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model.to(device)
        if cfg.distributed and cfg.sync_bn:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        if cfg.label_smoothing>0:
            criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
        else:
            criterion = nn.CrossEntropyLoss()
        model_without_ddp = model
        if cfg.distributed:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[cfg.gpu])
            model_without_ddp = model.module
        example_inputs = torch.randn(1, 3, 224, 224).to(device)
        special_type=cfg.model.split('_')[0]
        pruner = get_pruner(model, example_inputs, 0, "IMAGENET", cfg.method, special_type=special_type)
        base_ops, _ = tp.utils.count_ops_and_params(model, example_inputs=example_inputs)
        current_speed_up = 1.0
        acc1, acc5 = evaluate(model, criterion, test_loader, device)
        acc1_list = [acc1 / 100]
        acc5_list = [acc5 / 100]
        speed_up_list = [base_speed_up]
        while current_speed_up < max_speed_up / base_speed_up:
            pruner.step()
            if special_type == 'vit':
                model.module.hidden_dim = model.module.conv_proj.out_channels
            pruned_ops, _ = tp.utils.count_ops_and_params(model, example_inputs=example_inputs)
            acc1, acc5 = evaluate(model, criterion, test_loader, device)
            current_speed_up = float(base_ops) / pruned_ops
            print(f"Speed up: {current_speed_up*base_speed_up:.4f}")
            acc1_list.append(acc1 / 100)
            acc5_list.append(acc5 / 100)
            speed_up_list.append(current_speed_up * base_speed_up)
        del pruner
        return acc1_list, acc5_list, speed_up_list
    

    os.makedirs(save_dir, exist_ok=True)
    print("Start visualizing")
    plt.rcParams.update({
        'font.size': 12 * font_scale,           # General font size
        'axes.titlesize': 16 * font_scale,      # Title font size
        'axes.labelsize': 14 * font_scale,      # X and Y labels font size
        'xtick.labelsize': 12 * font_scale,     # X-axis tick labels
        'ytick.labelsize': 12 * font_scale,      # Y-axis tick labels
        'legend.fontsize': 12 * font_scale,      # Legend font size
        'figure.titlesize': 18 * font_scale      # Figure title size
    })
    plt.figure(figsize=(20, 20))
    if isinstance(models, list):
        assert isinstance(base_speed_up, list), 'if models are list, base_speed_up must be list !'
        for i, m in enumerate(models):
            acc1_list, acc5_list, speed_up_list = get_acc_speed_up_list(m, test_loader, base_speed_up[i], max_speed_up)
            plt.plot(speed_up_list, acc1_list, marker=marker, label=f"{labels[i]}_acc1", markersize=4*font_scale, linewidth=2*font_scale)
            plt.plot(speed_up_list, acc5_list, marker=marker, label=f"{labels[i]}_acc5", markersize=4*font_scale, linewidth=2*font_scale)
            print(f"Model {i+1}/{len(models)} visualized")
    else:
        acc1_list, acc5_list, speed_up_list = get_acc_speed_up_list(models, test_loader, base_speed_up, max_speed_up)
        plt.plot(speed_up_list, acc1_list, marker=marker, label=f"{labels}_acc1", markersize=4*font_scale, linewidth=2*font_scale)
        plt.plot(speed_up_list, acc5_list, marker=marker, label=f"{labels}_acc5", markersize=4*font_scale, linewidth=2*font_scale)
    plt.xlabel('Speed Up', fontsize=14 * font_scale)  # You can override individual elements if needed
    plt.ylabel('Test Acc', fontsize=14 * font_scale)
    plt.title('Test Acc vs. Speed Up', fontsize=16 * font_scale)
    plt.xlim(1.0, max_speed_up)
    plt.ylim(ylim)
    plt.locator_params(axis='y', nbins=20)
    plt.grid()
    plt.legend(loc='upper right', prop={'size': 12 * font_scale})
    plt.tick_params(axis='both', which='major', labelsize=12 * font_scale)
    plt.savefig(os.path.join(save_dir, name), dpi=300, bbox_inches='tight')
    plt.close()  # Close the figure to free memory
    print("End visualizing") 


@hydra.main(config_path="configs", config_name="base", version_base=None)
def main(cfg: DictConfig) -> None:
    if cfg.output_dir:
        utils.mkdir(cfg.output_dir)

    if not cfg.no_distribution:
        utils.init_distributed_mode(cfg)
    print(OmegaConf.to_yaml(cfg))

    device = torch.device(cfg.device)

    if cfg.use_deterministic_algorithms:
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.benchmark = True

    train_dir = os.path.join(cfg.data_path, "train")
    val_dir = os.path.join(cfg.data_path, "val")
    dataset, dataset_test, train_sampler, test_sampler = load_data(train_dir, val_dir, cfg)

    collate_fn = None
    num_classes = len(dataset.classes)
    mixup_transforms = []
    if cfg.mixup_alpha > 0.0:
        mixup_transforms.append(transforms.RandomMixup(num_classes, p=1.0, alpha=cfg.mixup_alpha))
    if cfg.cutmix_alpha > 0.0:
        mixup_transforms.append(transforms.RandomCutmix(num_classes, p=1.0, alpha=cfg.cutmix_alpha))
    if mixup_transforms:
        mixupcutmix = torchvision.transforms.RandomChoice(mixup_transforms)

        def collate_fn(batch):
            return mixupcutmix(*default_collate(batch))

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        sampler=train_sampler,
        num_workers=cfg.workers,
        pin_memory=cfg.pin_memory,
        collate_fn=collate_fn,
    )
    big_data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.big_batch_size,
        sampler=train_sampler,
        num_workers=cfg.workers,
        pin_memory=cfg.pin_memory,
        collate_fn=collate_fn,
    )
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=cfg.batch_size, sampler=test_sampler, num_workers=cfg.workers, pin_memory=cfg.pin_memory
    )
  
    if cfg.run == 'meta_train':
        metanetwork = hydra.utils.instantiate(cfg.metanetwork).to(device)
        if cfg.resume_epoch != -1:
            cfg.resume = os.path.join('save', f'{cfg.name}', 'meta_train', 'metanetwork', f"epoch_{cfg.resume_epoch}.pth")
        metanetwork = meta_train(cfg.data_model_num, metanetwork, cfg.meta_train.epochs, cfg.meta_train.lr, 
                                 cfg.meta_train.lr_warmup_epochs, train_sampler, big_data_loader, device, cfg.meta_train, cfg)


    elif cfg.run == 'prune': # no parallel
        ckpt = torch.load(os.path.join('save', f'{cfg.name}', f'{cfg.index}', 'train_from_scratch', 'latest.pth'), weights_only=False)
        model = state_dict_to_model(cfg.model, ckpt['model'])
        speed_up, model = progressive_pruning(model, 'IMAGENET', data_loader, data_loader_test, cfg.speed_up, cfg.method, log=True, eval_train_data=False, special_type=cfg.model.split('_')[0])
        torch.save(model.state_dict(), os.path.join(cfg.output_dir, f'{speed_up:.4f}.pth'))
        print(model)
    
    elif cfg.run == 'finetune': 
        dir = os.path.join('save', f'{cfg.name}', f'{cfg.index}', 'prune')
        pth_files = list(Path(dir).glob('*.pth'))
        assert len(pth_files) == 1, f'Only one pth file is expected in {dir}, but got {len(pth_files)}'
        state_dict = torch.load(pth_files[0], map_location=device, weights_only=False)
        model = state_dict_to_model(cfg.model, state_dict, device)
        del state_dict
        if cfg.resume_epoch != -1:
            cfg.resume = os.path.join('save', f'{cfg.name}', f'{cfg.index}', 'finetune', f'epoch_{cfg.resume_epoch}.pth')
        train(
            model,
            cfg.epochs,
            lr=cfg.lr,
            lr_warmup_epochs=cfg.lr_warmup_epochs,
            train_sampler=train_sampler,
            data_loader=data_loader,
            data_loader_test=data_loader_test,
            device=device,
            cfg=cfg,
            pruner=None,
            state_dict_only=True,
            save_every_epoch=True,
        )
        
    elif cfg.run == "visualize_origin":
        model_name = cfg.model
        savedir = os.path.join('save', f'{cfg.name}', 'visualize_origin', f'{cfg.index}')
        ckpt = torch.load(os.path.join('save', f'{cfg.name}', 'meta_train', 'data_model', f'{cfg.index}.pth'), weights_only=False, map_location=device)
        origin_state_dict = ckpt['model']
        origin_model = state_dict_to_model(model_name, origin_state_dict, device)
        visualize_acc_speed_up_curve(origin_model, 'origin', data_loader_test, 1.0, cfg, max_speed_up=2.5,
                                     save_dir=savedir, name=f'origin_visualize.png', ylim=(0.0, 1.0))   
        
    elif cfg.run == 'visualize':
        savedir = os.path.join('save', f'{cfg.name}', 'visualize', f'{cfg.index}', f'metanetwork_{cfg.metanetwork_index}')
        print('save dir : ', savedir)
        os.makedirs(savedir, exist_ok=True)
        model_name = cfg.model
        example_inputs = torch.randn(1, 3, 224, 224).to(device)
        base_model = registry.get_model(num_classes=1000, name=cfg.model, pretrained=cfg.pretrained, target_dataset='imagenet').to(device)
        base_ops, base_params = tp.utils.count_ops_and_params(base_model, example_inputs=example_inputs)
        del base_model

        
        ckpt = torch.load(os.path.join('save', f'{cfg.name}', 'meta_train', 'data_model', f'{cfg.index}.pth'), weights_only=False, map_location=device)
        origin_state_dict = ckpt['model']
        origin_model = state_dict_to_model(model_name, origin_state_dict, device)

        if cfg.resume_epoch == -1:
            metanetwork = torch.load(os.path.join('save', f'{cfg.name}', 'meta_train', 'metanetwork', f"epoch_{cfg.metanetwork_index}.pth"), weights_only=False, map_location=device)['model']
            print(f'load metanetwork from {os.path.join("save", f"{cfg.name}", "meta_train", "metanetwork", f"epoch_{cfg.metanetwork_index}.pth")}')
            node_index, node_features, edge_index, edge_features_list = state_dict_to_graph(model_name, origin_state_dict, device)
            node_pred, edge_pred = metanetwork.forward(node_features, edge_index, edge_features_list)
            state_dict = graph_to_state_dict(model_name, origin_state_dict, node_index, node_pred, edge_index, edge_pred, device)
            model = state_dict_to_model(cfg.model, state_dict, device)
            del state_dict, node_index, node_features, edge_index, edge_features_list, node_pred, edge_pred
            pruned_ops, pruned_params = tp.utils.count_ops_and_params(model, example_inputs=example_inputs)
        else:
            cfg.resume = os.path.join('save', f'{cfg.name}', 'visualize', f'{cfg.index}', f'metanetwork_{cfg.metanetwork_index}', f'epoch_{cfg.resume_epoch}.pth')
            ckpt = torch.load(cfg.resume, weights_only=False, map_location=device)
            model = state_dict_to_model(cfg.model, ckpt['model'], device)
            pruned_ops, pruned_params = tp.utils.count_ops_and_params(model, example_inputs=example_inputs)


        print(f"Origin speed up : {base_ops / pruned_ops}")
        cfg.output_dir = savedir
        model = train(model, cfg.epochs, cfg.lr, cfg.lr_warmup_epochs, train_sampler, data_loader, data_loader_test, device, cfg, pruner=None, state_dict_only=True, save_every_epoch=True)
        
        model_list = [model, origin_model]
        label_list = ['after metanetwork', 'origin']
        speed_up_list = [base_ops / pruned_ops, base_ops / pruned_ops]
        visualize_acc_speed_up_curve(model_list, label_list, data_loader_test, speed_up_list, cfg, max_speed_up=2.5,
                                     save_dir=savedir, name=f'visualize.png', ylim=(0.6, 1.0))   
            

    elif cfg.run == 'prune_after_metanetwork':
        savedir = os.path.join('save', f'{cfg.name}', 'prune_after_metanetwork', f'{cfg.index}', f'metanetwork_{cfg.metanetwork_index}', f'{cfg.speed_up:.4f}')
        print('save dir : ', savedir)
        os.makedirs(savedir, exist_ok=True)
        model_name = cfg.model
        if cfg.resume_epoch == -1:
            example_inputs = torch.randn(1, 3, 224, 224).to(device)
            base_model = registry.get_model(num_classes=1000, name=cfg.model, pretrained=cfg.pretrained, target_dataset='imagenet').to(device)
            base_ops, base_params = tp.utils.count_ops_and_params(base_model, example_inputs=example_inputs)
            del base_model
            ckpt = torch.load(os.path.join('save', f'{cfg.name}', 'visualize', f'{cfg.index}', f'metanetwork_{cfg.metanetwork_index}', 'best.pth'), weights_only=False, map_location=device)
            model = state_dict_to_model(model_name, ckpt['model'], device)
            pruned_ops, pruned_params = tp.utils.count_ops_and_params(model, example_inputs=example_inputs)
            current_speed_up = float(base_ops) / pruned_ops
            print(f'Origin speed up : {current_speed_up}')
            speed_up, model = progressive_pruning(model, 'IMAGENET', data_loader, data_loader_test, cfg.speed_up / current_speed_up, cfg.method, log=True, eval_train_data=False, eval_test_data=False, special_type=cfg.model.split('_')[0])
            current_speed_up *= speed_up
            print(f"Current speed up : {current_speed_up}")
        else:
            ckpt = torch.load(os.path.join('save', f'{cfg.name}', 'prune_after_metanetwork', f'{cfg.index}', f'metanetwork_{cfg.metanetwork_index}', f'{cfg.speed_up}', f'epoch_{cfg.resume_epoch}.pth'), weights_only=False, map_location=device)
            model = state_dict_to_model(model_name, ckpt['model'], device)
            cfg.resume = os.path.join('save', f'{cfg.name}', 'prune_after_metanetwork', f'{cfg.index}', f'metanetwork_{cfg.metanetwork_index}', f'{cfg.speed_up}', f'epoch_{cfg.resume_epoch}.pth')

        cfg.lr_decay_milestones = "120,160,185"
        cfg.output_dir = savedir
        model = train(model, 200, 0.01, cfg.lr_warmup_epochs, train_sampler, data_loader, data_loader_test, device, cfg, pruner=None, state_dict_only=True, save_every_epoch=True)


    elif cfg.run == 'train_from_scratch':
        if cfg.rank == 0:
            model = registry.get_model(num_classes=1000, name=cfg.model, pretrained=cfg.pretrained, target_dataset='imagenet').to(device)
            dist.barrier()
        else:
            dist.barrier()
            model = registry.get_model(num_classes=1000, name=cfg.model, pretrained=cfg.pretrained, target_dataset='imagenet').to(device)
        train(
            model,
            cfg.epochs,
            lr=cfg.lr,
            lr_warmup_epochs=cfg.lr_warmup_epochs,
            train_sampler=train_sampler,
            data_loader=data_loader,
            data_loader_test=data_loader_test,
            device=device,
            cfg=cfg,
            pruner=None,
            state_dict_only=True,
            save_every_epoch=True,
        )


    
    elif cfg.run == 'test':
        model.to(device)
        if cfg.distributed and cfg.sync_bn:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

        if cfg.label_smoothing>0:
            criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
        else:
            criterion = nn.CrossEntropyLoss()

        # scaler = torch.cuda.amp.GradScaler() if args.amp else None
        scaler = torch.amp.GradScaler('cuda') if cfg.amp else None
        model_without_ddp = model
        if cfg.distributed:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[cfg.gpu])
            model_without_ddp = model.module

        if cfg.resume:
            checkpoint = torch.load(cfg.resume, map_location="cpu")
            model_without_ddp.load_state_dict(checkpoint["model"])

        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        evaluate(model, criterion, data_loader_test, device=device)


    else:
        raise ValueError(f"Run type {cfg.run} not supported.")

    if cfg.distributed:
        torch.distributed.destroy_process_group()

if __name__ == "__main__":
    main()