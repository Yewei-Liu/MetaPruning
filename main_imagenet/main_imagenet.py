import datetime
import os
import sys
import time
import warnings
import registry

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
from torchvision.transforms.functional import InterpolationMode

import torch_pruning as tp
from functools import partial
from omegaconf import DictConfig, OmegaConf
import hydra


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


def get_pruner(model, example_inputs, cfg):
    unwrapped_parameters = (
        [model.encoder.pos_embedding, model.class_token] if "vit" in cfg.model else None
    )
    sparsity_learning = False
    data_dependency = False
    if cfg.method == "random":
        imp = tp.importance.RandomImportance()
        pruner_entry = partial(tp.pruner.MagnitudePruner, global_pruning=cfg.global_pruning)
    elif cfg.method == "l1":
        imp = tp.importance.MagnitudeImportance(p=1)
        pruner_entry = partial(tp.pruner.MagnitudePruner, global_pruning=cfg.global_pruning)
    elif cfg.method == "lamp":
        imp = tp.importance.LAMPImportance(p=2)
        pruner_entry = partial(tp.pruner.MagnitudePruner, global_pruning=cfg.global_pruning)
    elif cfg.method == "slim":
        sparsity_learning = True
        imp = tp.importance.BNScaleImportance()
        pruner_entry = partial(tp.pruner.BNScalePruner, reg=cfg.reg, global_pruning=cfg.global_pruning)
    elif cfg.method == "group_norm":
        imp = tp.importance.GroupMagnitudeImportance(p=2)
        pruner_entry = partial(tp.pruner.GroupNormPruner, global_pruning=cfg.global_pruning)
    elif cfg.method == "group_greg":
        sparsity_learning = True
        imp = tp.importance.GroupMagnitudeImportance(p=2)
        pruner_entry = partial(
            tp.pruner.GrowingRegPruner,
            reg=cfg.reg,
            delta_reg=cfg.delta_reg,
            global_pruning=cfg.global_pruning,
        )
    elif cfg.method == "group_sl":
        sparsity_learning = True
        imp = tp.importance.GroupMagnitudeImportance(p=2)
        pruner_entry = partial(tp.pruner.GroupNormPruner, reg=cfg.reg, global_pruning=cfg.global_pruning)
    else:
        raise NotImplementedError
    cfg.data_dependency = data_dependency
    cfg.sparsity_learning = sparsity_learning
    ignored_layers = []
    pruning_ratio_dict = {}
    for m in model.modules():
        if isinstance(m, torch.nn.Linear) and m.out_features == 1000:
            ignored_layers.append(m)
    round_to = None
    if "vit" in cfg.model:
        round_to = model.encoder.layers[0].num_heads
    pruner = pruner_entry(
        model,
        example_inputs,
        importance=imp,
        iterative_steps=100,
        pruning_ratio=1.0,
        pruning_ratio_dict=pruning_ratio_dict,
        max_pruning_ratio=cfg.max_pruning_ratio,
        ignored_layers=ignored_layers,
        round_to=round_to,
        unwrapped_parameters=unwrapped_parameters,
    )
    return pruner


def train_one_epoch(
    model, criterion, optimizer, data_loader, device, epoch, cfg, model_ema=None, scaler=None, pruner=None, recover=None
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

        if model_ema and i % cfg.model_ema_steps == 0:
            model_ema.update_parameters(model)
            if epoch < cfg.lr_warmup_epochs:
                model_ema.n_averaged.fill_(0)

        acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
        batch_size = image.shape[0]
        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
        metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
        metric_logger.meters["img/s"].update(batch_size / (time.time() - start_time))
    if pruner is not None and isinstance(pruner, tp.pruner.GrowingRegPruner):
        pruner.update_reg()


def evaluate(model, criterion, data_loader, device, print_freq=100, log_suffix=""):
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
    return metric_logger.acc1.global_avg


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
    lr, lr_step_size, lr_warmup_epochs, 
    train_sampler, data_loader, data_loader_test, 
    device, cfg, pruner=None, state_dict_only=True, recover=None):

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
    scaler = torch.amp.GradScaler("cuda") if cfg.amp else None

    cfg.lr_scheduler = cfg.lr_scheduler.lower()
    if cfg.lr_scheduler == "steplr":
        main_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_step_size, gamma=cfg.lr_gamma)
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

    model_ema = None
    if cfg.model_ema:
        # Decay adjustment that aims to keep the decay independent from other hyper-parameters originally proposed at:
        # https://github.com/facebookresearch/pycls/blob/f8cd9627/pycls/core/net.py#L123
        #
        # total_ema_updates = (Dataset_size / n_GPUs) * epochs / (batch_size_per_gpu * EMA_steps)
        # We consider constant = Dataset_size for a given dataset/setup and ommit it. Thus:
        # adjust = 1 / total_ema_updates ~= n_GPUs * batch_size_per_gpu * EMA_steps / epochs
        adjust = cfg.world_size * cfg.batch_size * cfg.model_ema_steps / epochs
        alpha = 1.0 - cfg.model_ema_decay
        alpha = min(1.0, alpha * adjust)
        model_ema = utils.ExponentialMovingAverage(model_without_ddp, device=device, decay=1.0 - alpha)

    if cfg.resume:
        checkpoint = torch.load(cfg.resume, map_location="cpu")
        model_without_ddp.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        cfg.start_epoch = checkpoint["epoch"] + 1
        if model_ema:
            model_ema.load_state_dict(checkpoint["model_ema"])
        if scaler:
            scaler.load_state_dict(checkpoint["scaler"])
    
    start_time = time.time()
    best_acc = 0
    prefix = '' if pruner is None else 'regularized_{:e}_'.format(cfg.reg)
    for epoch in range(cfg.start_epoch, epochs):
        if cfg.distributed:
            train_sampler.set_epoch(epoch)
        train_one_epoch(model, criterion, optimizer, data_loader, device, epoch, cfg, model_ema, scaler, pruner, recover=recover)
        lr_scheduler.step()
        acc = evaluate(model, criterion, data_loader_test, device=device)
        if model_ema:
            acc = evaluate(model_ema, criterion, data_loader_test, device=device, log_suffix="EMA")
        if cfg.output_dir:
            checkpoint = {
                "model": model_without_ddp.state_dict() if state_dict_only else model_without_ddp,
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "epoch": epoch,
                "args": cfg,
            }
            if model_ema:
                checkpoint["model_ema"] = model_ema.state_dict()
            if scaler:
                checkpoint["scaler"] = scaler.state_dict()
            if acc>best_acc:
                best_acc=acc
                utils.save_on_master(checkpoint, os.path.join(cfg.output_dir, prefix+"best.pth"))
            utils.save_on_master(checkpoint, os.path.join(cfg.output_dir, prefix+"latest.pth"))
        print("Epoch {}/{}, Current Best Acc = {:.6f}".format(epoch, epochs, best_acc))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Training time {total_time_str}")
    if cfg.distributed:
        torch.distributed.destroy_process_group()
    return model_without_ddp


@hydra.main(config_path="configs", config_name="base", version_base=None)
def main(cfg: DictConfig) -> None:
    if cfg.output_dir:
        utils.mkdir(cfg.output_dir)

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
        pin_memory=True,
        collate_fn=collate_fn,
    )
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=cfg.batch_size, sampler=test_sampler, num_workers=cfg.workers, pin_memory=True
    )

    print("Creating model")
    model = registry.get_model(num_classes=1000, name=cfg.model, pretrained=cfg.pretrained, target_dataset="imagenet")
    model.eval()
    print("="*16)
    print(model)
    example_inputs = torch.randn(1, 3, 224, 224)
    base_ops, base_params = tp.utils.count_ops_and_params(model, example_inputs=example_inputs)
    print("Params: {:.4f} M".format(base_params / 1e6))
    print("Ops: {:.4f} G".format(base_ops / 1e9))
    print("="*16)

    if cfg.run == 'prune_sl':
        pruner = get_pruner(model, example_inputs=example_inputs, cfg=cfg)
        if cfg.sparsity_learning:
            if cfg.sl_resume:
                print("Loading sparse model from {}...".format(cfg.sl_resume))
                model.load_state_dict(torch.load(cfg.sl_resume, map_location="cpu")["model"])
            else:
                print("Sparsifying model...")
                sl_lr = cfg.sl_lr if cfg.sl_lr is not None else cfg.lr
                sl_lr_step_size = cfg.sl_lr_step_size if cfg.sl_lr_step_size is not None else cfg.lr_step_size
                sl_lr_warmup_epochs = cfg.sl_lr_warmup_epochs if cfg.sl_lr_warmup_epochs is not None else cfg.lr_warmup_epochs
                sl_epochs = cfg.sl_epochs if cfg.sl_epochs is not None else cfg.epochs
                train(
                    model,
                    sl_epochs,
                    lr=sl_lr,
                    lr_step_size=sl_lr_step_size,
                    lr_warmup_epochs=sl_lr_warmup_epochs,
                    train_sampler=train_sampler,
                    data_loader=data_loader,
                    data_loader_test=data_loader_test,
                    device=device,
                    cfg=cfg,
                    pruner=pruner,
                    state_dict_only=True,
                )

        model = model.to("cpu")
        print("Pruning model...")
        prune_to_target_flops(pruner, model, cfg.target_flops, example_inputs, cfg)
        pruned_ops, pruned_size = tp.utils.count_ops_and_params(model, example_inputs=example_inputs)
        print("="*16)
        print("After pruning:")
        print(model)
        print("Params: {:.2f} M => {:.2f} M ({:.2f}%)".format(base_params / 1e6, pruned_size / 1e6, pruned_size / base_params * 100))
        print("Ops: {:.2f} G => {:.2f} G ({:.2f}%, {:.2f}X )".format(base_ops / 1e9, pruned_ops / 1e9, pruned_ops / base_ops * 100, base_ops / pruned_ops))
        print("="*16)

        dataset, dataset_test, train_sampler, test_sampler = load_data(train_dir, val_dir, cfg)
        print("Finetuning...")
        train(
            model,
            cfg.epochs,
            lr=cfg.lr,
            lr_step_size=cfg.lr_step_size,
            lr_warmup_epochs=cfg.lr_warmup_epochs,
            train_sampler=train_sampler,
            data_loader=data_loader,
            data_loader_test=data_loader_test,
            device=device,
            cfg=cfg,
            pruner=None,
            state_dict_only=False,
        )
        
    elif cfg.run == 'prune':
        pruner = get_pruner(model, example_inputs=example_inputs, cfg=cfg)
        model = model.to("cpu")
        print("Pruning model...")
        prune_to_target_flops(pruner, model, cfg.target_flops, example_inputs, cfg)
        pruned_ops, pruned_size = tp.utils.count_ops_and_params(model, example_inputs=example_inputs)
        print("="*16)
        print("After pruning:")
        print(model)
        print("Params: {:.2f} M => {:.2f} M ({:.2f}%)".format(base_params / 1e6, pruned_size / 1e6, pruned_size / base_params * 100))
        print("Ops: {:.2f} G => {:.2f} G ({:.2f}%, {:.2f}X )".format(base_ops / 1e9, pruned_ops / 1e9, pruned_ops / base_ops * 100, base_ops / pruned_ops))
        print("="*16)

        dataset, dataset_test, train_sampler, test_sampler = load_data(train_dir, val_dir, cfg)
        print("Finetuning...")
        train(
            model,
            cfg.epochs,
            lr=cfg.lr,
            lr_step_size=cfg.lr_step_size,
            lr_warmup_epochs=cfg.lr_warmup_epochs,
            train_sampler=train_sampler,
            data_loader=data_loader,
            data_loader_test=data_loader_test,
            device=device,
            cfg=cfg,
            pruner=None,
            state_dict_only=False,
        )

    elif cfg.run == 'train':
        train(
            model,
            cfg.epochs,
            lr=cfg.lr,
            lr_step_size=cfg.lr_step_size,
            lr_warmup_epochs=cfg.lr_warmup_epochs,
            train_sampler=train_sampler,
            data_loader=data_loader,
            data_loader_test=data_loader_test,
            device=device,
            cfg=cfg,
            pruner=None,
            state_dict_only=True,
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
        scaler = torch.amp.GradScaler("cuda") if cfg.amp else None
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

        if cfg.distributed:
            torch.distributed.destroy_process_group()



    else:
        raise ValueError(f"Run type {cfg.run} not supported.")

if __name__ == "__main__":
    main()