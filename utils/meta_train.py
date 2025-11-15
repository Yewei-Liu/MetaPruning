import torch
import torch.nn.functional as F
from torch.func import functional_call
import numpy as np
from utils.mylogging import get_logger
from utils.train import train, eval
from utils.convert import graph_to_state_dict, state_dict_to_model, graph_to_model
from utils.pruner import get_pruner
from utils.pruning import adaptive_pruning
from generate_dataset.resnet_family import resnet56
import omegaconf
from copy import deepcopy
import os


def meta_eval(
    model_name,
    dataset_name,
    model_loader,
    big_train_data_loader,
    small_train_data_loader,
    big_val_data_loader,
    metanetwork,
    method,
    cfg,
    log = True,
    device = None,
    verbose = False,
):
    '''
    Evaluation the quality of metanetwork during meta-training.
    We didn't use it in our final experiments because it costs too much time (need finetuning during each eval).
    You can explore it by yourself if you want.
    '''
    
    speed_up_threshold = cfg.speed_up_threshold
    epochs = cfg.epochs
    lr = cfg.lr
    lr_decay_milestones = cfg.lr_decay_milestones
    lr_decay_gamma = cfg.lr_decay_gamma
    weight_decay = cfg.weight_decay

    train_acc_list = []
    train_loss_list = []
    val_acc_list = []
    val_loss_list = []
    speed_up_list = []
    gt_train_acc_list = []
    gt_train_loss_list = []
    gt_val_acc_list = []
    gt_val_loss_list = []
    gt_speed_up_list = []
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if log:
        logger = get_logger(f"meta eval")
    metanetwork.eval().to(device)
    for i, (model, origin_state_dict, info, node_index, node_features, edge_index, edge_features) in enumerate(model_loader):  
        gt_train_acc_list.append(info['train_acc'])
        gt_train_loss_list.append(info['train_loss'])
        gt_val_acc_list.append(info['val_acc'])
        gt_val_loss_list.append(info['val_loss'])
        gt_speed_up_list.append(info['current_speed_up'])
        tmp_model = state_dict_to_model(model_name, model.state_dict())
        tmp_model.eval().to(device)
        if log and verbose:
            logger.info("Origin :")
        with torch.no_grad():
            node_pred, edge_pred = metanetwork.forward(node_features.to(device), 
                                                       [ei.to(device) for ei in edge_index] if isinstance(edge_index, list) else edge_index.to(device), 
                                                       [ef.to(device) for ef in edge_features] if isinstance(edge_features, list) else edge_features.to(device))
            model = graph_to_model(model_name, origin_state_dict, node_index, node_pred, edge_index, edge_pred, device)
        train(model, small_train_data_loader, big_val_data_loader, epochs, lr, lr_decay_milestones, 
              lr_decay_gamma, weight_decay, log=verbose, return_best=True)
        train_acc, train_loss = eval(model, big_train_data_loader, device)
        val_acc, val_loss = eval(model, big_val_data_loader, device)
        train_acc_list.append(train_acc)
        train_loss_list.append(train_loss)
        val_acc_list.append(val_acc)
        val_loss_list.append(val_loss)
        if log and verbose:
            logger.info("Metanetwork + finetune :")
        speedup, _ = adaptive_pruning(model, model_name, dataset_name, big_train_data_loader, big_val_data_loader, speed_up_threshold, method, verbose, device)
        speed_up_list.append(speedup)

    res = {'train_acc': train_acc_list, 'train_loss': train_loss_list, 
           'val_acc': val_acc_list, 'val_loss': val_loss_list, 
           'speed_up' : speed_up_list,
           'gt_train_acc': gt_train_acc_list, 'gt_train_loss': gt_train_loss_list,
           'gt_val_acc': gt_val_acc_list, 'gt_val_loss': gt_val_loss_list,
           'gt_speed_up' :  gt_speed_up_list}
    if log :
        logger.info(
            f"meta eval res :\n\
            Train acc={np.mean(train_acc_list)}({np.std(train_acc_list)})\n\
            Gt Train acc={np.mean(gt_train_acc_list)}({np.std(gt_train_acc_list)})\n\
            Train Loss={np.mean(train_loss_list)}({np.std(train_loss_list)})\n\
            Gt Train loss={np.mean(gt_train_loss_list)}({np.std(gt_train_loss_list)})\n\
            Val Acc={np.mean(val_acc_list)}({np.std(val_acc_list)})\n\
            Gt Val Acc={np.mean(gt_val_acc_list)}({np.std(gt_val_acc_list)})\n\
            Val Loss={np.mean(val_loss_list)}({np.std(val_loss_list)})\n\
            Gt Val Loss={np.mean(gt_val_loss_list)}({np.std(gt_val_loss_list)})\n\
            Speed up={np.mean(speed_up_list)}({np.std(speed_up_list)})\n\
            Gt Speed up={np.mean(gt_speed_up_list)}({np.std(gt_speed_up_list)})"
        )
    return res


def meta_train(
    metanetwork,
    model_train_loader,
    big_data_train_loader,
    small_data_train_loader,
    cfg,
    log=True,
    model_val_loader=None,
    big_data_val_loader=None,
):
    epochs = cfg.epochs
    lr = cfg.lr
    lr_decay_milestones = cfg.lr_decay_milestones
    lr_decay_gamma = cfg.lr_decay_gamma
    weight_decay = cfg.weight_decay
    method = cfg.method
    alpha = cfg.alpha
    bias = cfg.bias
    pruner_reg = cfg.pruner_reg
    save_every_epoch = cfg.save_every_epoch
    warm_up = cfg.warm_up
    save_path = cfg.save_path
    model_name = cfg.model_name
    use_meta_eval = cfg.use_meta_eval # meta eval takes longer time
    eval_cfg = cfg.meta_eval
    dataset_name = cfg.dataset_name

    os.makedirs(save_path, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optimizer = torch.optim.AdamW(
        metanetwork.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )
    milestones = [int(ms) for ms in lr_decay_milestones.split(",")]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=milestones, gamma=lr_decay_gamma
    )
    if log:
        logger = get_logger("meta_train")
    metanetwork.eval().to(device)

    for epoch in range(epochs):
        metanetwork.train()
        for i, (model, origin_state_dict, info, node_index, node_features, edge_index, edge_features) in enumerate(model_train_loader):
            def onetrainstep():
                model.to(device)
                pruner = get_pruner(model, torch.ones((1, 3, 32, 32)).to(device), pruner_reg, dataset_name, method=method)
                node_pred, edge_pred = metanetwork.forward(node_features.to(device), 
                                                           [ei.to(device) for ei in edge_index] if isinstance(edge_index, list) else edge_index.to(device), 
                                                           [ef.to(device) for ef in edge_features] if isinstance(edge_features, list) else edge_features.to(device))
                state_dict = graph_to_state_dict(model_name, origin_state_dict, node_index, node_pred, edge_index, edge_pred, device)
                losses = 0.0
                optimizer.zero_grad()
                for j, (data, target) in enumerate(big_data_train_loader):
                    data, target = data.to(device), target.to(device)   
                    out = functional_call(model, state_dict, data)
                    loss = F.cross_entropy(out, target)
                    losses += loss.item()
                    loss.backward(retain_graph=True)
                model.load_state_dict(state_dict)
                for param in model.parameters():
                    if param.requires_grad:
                        param.grad = torch.zeros_like(param)    
                pruner.regularize(model, alpha=alpha, bias=bias)
                big_tensor = []
                big_gradient = []
                for name, param in model.named_parameters():
                    if param.requires_grad:
                        if not torch.isnan(param.grad).any():
                            big_tensor.append(state_dict[name].flatten())
                            big_gradient.append(param.grad.flatten().detach().clone())
                big_tensor = torch.cat(big_tensor, dim=0)
                big_gradient = torch.cat(big_gradient, dim=0)
                big_tensor.backward(gradient=big_gradient)
                optimizer.step()
                del pruner
                
                if log:
                    logger.info(
                        "Epoch {:d}/{:d}, iter {:d}/{:d}, train loss={:.4f}, lr={}".format(
                            epoch + 1,
                            epochs,
                            i + 1,
                            len(model_train_loader),
                            losses / len(big_data_train_loader),
                            optimizer.param_groups[0]["lr"],
                        )
                    )
            onetrainstep()
        scheduler.step()
        if ((epoch + 1) % save_every_epoch == 0 or epoch == epochs - 1) and epoch >= warm_up:
            if use_meta_eval:
                assert model_val_loader is not None, "Need model_val_loader to meta eval !"
                assert big_data_val_loader is not None, "Need data_val_loader to meta eval !"
                res = meta_eval(
                    model_name, dataset_name, model_val_loader, big_data_train_loader, small_data_train_loader, big_data_val_loader, 
                    metanetwork, method, eval_cfg, log, device
                )
                speed_up = np.mean(res['speed_up'])
                torch.save(metanetwork, os.path.join(save_path, f'epoch_{epoch + 1}_{speed_up:.3f}.pth'))
            else:
                torch.save(metanetwork, os.path.join(save_path, f'epoch_{epoch + 1}.pth'))
            if log:
                logger.info(f"epoch_{epoch + 1} saved !")
    
    return metanetwork