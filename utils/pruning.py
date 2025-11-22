from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_pruning as tp
from copy import deepcopy
from utils.mylogging import get_logger
from utils.convert import state_dict_to_model, state_dict_to_graph, graph_to_state_dict, graph_to_model
from utils.train import train, eval
from utils.pruner import get_pruner
from utils.unstructural_flops import count_model_flops_and_params

def progressive_pruning(
        model, 
        dataset_name,
        train_loader,
        test_loader,
        speed_up, 
        method,
        log=False, 
        device=None,
        eval_train_data=True,
        eval_test_data=True,
        special_type=None,
        return_params=False,
):
    '''
    Pruning to a fixed speed up.
    '''
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.eval().to(device)
    if dataset_name.lower() in ['cifar10', 'cifar100', 'svhn']:
        example_inputs = torch.ones((1, 3, 32, 32)).to(device)
    elif dataset_name.lower() in ['imagenet', 'cifar10(224)']:
        example_inputs = torch.ones((1, 3, 224, 224)).to(device)
    elif dataset_name.lower() in ['voc07']:
        example_inputs = torch.ones((1, 3, 1000, 800)).to(device)
    else:
        raise NotImplementedError(f"Dataset {dataset_name} is not supported.")
    pruner = get_pruner(model, example_inputs, 0.1, dataset_name, method=method, special_type=special_type)
    base_ops, base_params = tp.utils.count_ops_and_params(model, example_inputs=example_inputs)
    current_speed_up = 1.0
    if log:
        logger = get_logger("Progressive pruning")
    pruned_params = None
    while current_speed_up < speed_up :
        pruner.step()
        if special_type == 'vit':
            model.hidden_dim = model.conv_proj.out_channels
        pruned_ops, pruned_params = tp.utils.count_ops_and_params(model, example_inputs=example_inputs)
        current_speed_up = float(base_ops) / pruned_ops
        if pruner.current_step == pruner.iterative_steps:
            break
    del pruner
    if eval_train_data:
        train_acc, train_loss = eval(model, train_loader, device=device)
    if eval_test_data:
        val_acc, val_loss = eval(model, test_loader, device=device)
    if log:
        if eval_train_data:
            logger.info(f'Train acc : {train_acc}')
        if eval_test_data:
            logger.info(f"Val acc : {val_acc}")
        logger.info(f"Current speed up: {current_speed_up:.2f}")
    if return_params:
        return current_speed_up, model, base_params, pruned_params
    else:
        return current_speed_up, model


def adaptive_pruning(
        
        model, 
        model_name,
        dataset_name,
        train_loader, 
        test_loader, 
        acc_threshold=0.9, 
        method = None,
        log=False, 
        device=None,
        target_speed_up=float('inf')):
    '''
    Pruning until the accuracy reaches a threshold.
    '''
    
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.eval().to(device)
    example_inputs = torch.ones((1, 3, 32, 32)).to(device)
    pruner = get_pruner(model, example_inputs, 0.1, dataset_name, method=method)
    base_ops, _ = tp.utils.count_ops_and_params(model, example_inputs=example_inputs)
    current_speed_up = 1
    cur_model_state_dict = deepcopy(model.state_dict())
    if log:
        logger = get_logger("Adaptive pruning")
    while True:
        pruner.step()
        pruned_ops, _ = tp.utils.count_ops_and_params(model, example_inputs=example_inputs)
        acc, loss = eval(model, test_loader, device)
        if acc < acc_threshold:
            break
        cur_model_state_dict = deepcopy(model.state_dict())
        current_speed_up = float(base_ops) / pruned_ops
        if log:
            logger.info(f"Val acc : {acc}   Current speed up: {current_speed_up:.2f}")
        if pruner.current_step == pruner.iterative_steps or current_speed_up >= target_speed_up:
            break
    del pruner
    model = state_dict_to_model(model_name, cur_model_state_dict)
    model.eval().to(device)
    train_acc, train_loss = eval(model, train_loader, device=device)
    val_acc, val_loss = eval(model, test_loader, device=device)
    if log:
        logger.info(f"Train acc : {train_acc}   Val acc : {val_acc}   Current speed up: {current_speed_up:.2f}")
    return current_speed_up, model


def pruning_one_step(model, 
                     model_name,
                     dataset_name, 
                     info,
                     origin_state_dict,
                     big_train_loader, 
                     small_train_loader,
                     big_test_loader, 
                     metanetwork, # If metanetwork = -1, No metanetwork
                     cfg,
                     current_speed_up=1.0, 
                     log=False, 
                     device=None,
):
    # Pruning_index < 1.0 : adaptive_pruning. Otherwise >= 1.0 : progressive_pruning.
    pruning_index = cfg.pruning_index 
    method = cfg.method

    def train_with_cfg(model, cfg, opt, return_best):
        return train(model, small_train_loader, big_test_loader, cfg.epochs, cfg.lr, cfg.lr_decay_milestones, cfg.lr_decay_gamma, 
                     cfg.weight_decay, log=log, return_best=return_best, opt=opt)
    def finetune(model, mode):
        if mode == 'after pruning':
            train_acc, train_loss, val_acc, val_loss = train_with_cfg(model, cfg.finetune.after_pruning, cfg.opt, True)
            return train_acc, train_loss, val_acc, val_loss
        elif mode == 'after metanetwork':
            train_acc, train_loss, val_acc, val_loss = train_with_cfg(model, cfg.finetune.after_metanetwork, cfg.opt, False)
            return train_acc, train_loss, val_acc, val_loss
        else:
            raise NotImplementedError

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.eval().to(device)
    if log:
        logger = get_logger("Pruning")
    final_model = state_dict_to_model(model_name, origin_state_dict).eval().to(device)
    node_index, node_features, edge_index, edge_features = state_dict_to_graph(model_name, origin_state_dict)
    if metanetwork != -1:
        metanetwork.eval().to(device)
        node_pred, edge_pred = metanetwork.forward(node_features.to(device), edge_index.to(device), edge_features.to(device))
        model = graph_to_model(model_name, origin_state_dict, node_index, node_pred, edge_index, edge_pred)
        train_acc, train_loss, val_acc, val_loss = finetune(model, mode='after metanetwork')
    else:
        node_pred, edge_pred = deepcopy(node_features), deepcopy(edge_features)
        model = graph_to_model(model_name, origin_state_dict, node_index, node_pred, edge_index, edge_pred)
        if log:
            logger.info("No metanetwork, use original node and edge features.")
    
    if pruning_index < 1.0:
        speed_up, model = adaptive_pruning(model, model_name, dataset_name, big_train_loader, big_test_loader, pruning_index, method, log=log)
    else:
        assert pruning_index > current_speed_up, f"pruning_index {pruning_index} should be larger than current speed up {current_speed_up}"
        speed_up, model, base_params, pruned_params = progressive_pruning(model, dataset_name, big_train_loader, big_test_loader, pruning_index / current_speed_up, 
                                                             method=method, log=log, return_params=True)
    train_acc, train_loss, val_acc, val_loss = finetune(model, mode='after pruning')
    final_model = state_dict_to_model(model_name, model.state_dict())
    current_speed_up *= speed_up
    if log:
        logger.info(f"\n{final_model}")
        logger.info(f"Origin val acc : {info['val_acc']} Final val acc : {val_acc}\n\
                      Speed up: {speed_up:.2f}   Final speed up: {current_speed_up:.2f}\
                      Origin params: {base_params}   Params left: {pruned_params}")

    return final_model

def unstructured_pruning(
        model, 
        dataset_name,
        train_loader,
        test_loader,
        pruning_amount,
        method,
        log=False, 
        device=None,
        eval_train_data=True,
        eval_test_data=True,
):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.eval().to(device)
    if dataset_name.lower() in ['cifar10', 'cifar100', 'svhn']:
        example_inputs = torch.ones((1, 3, 32, 32)).to(device)
    elif dataset_name.lower() in ['imagenet', 'cifar10(224)']:
        example_inputs = torch.ones((1, 3, 224, 224)).to(device)
    else:
        raise NotImplementedError(f"Dataset {dataset_name} is not supported.")
    if log:
        logger = get_logger("Unstructured pruning")
    if isinstance(pruning_amount, str):
        pruner = get_pruner(model, example_inputs, 0.1, dataset_name, method="nmsparsity")
        pruner.step(pruning_amount)
    else:
        pruner = get_pruner(model, example_inputs, 0.1, dataset_name, method=method)
        pruner.step(pruning_amount)
    base_ops, left_ops, base_params, left_params, _, _ = count_model_flops_and_params(model, example_inputs, verbose=False)
    del pruner
    if eval_train_data:
        train_acc, train_loss = eval(model, train_loader, device=device)
    if eval_test_data:
        val_acc, val_loss = eval(model, test_loader, device=device)
    if log:
        if eval_train_data:
            logger.info(f'Train acc : {train_acc}')
        if eval_test_data:
            logger.info(f"Val acc : {val_acc}")
        logger.info(f"Pruned params: {base_params - left_params} / {base_params}")
    return model, base_params, left_params

def unstructured_pruning_one_step(
    model, 
    model_name,
    dataset_name, 
    info,
    origin_state_dict,
    big_train_loader, 
    small_train_loader,
    big_test_loader, 
    metanetwork, # If metanetwork = -1, No metanetwork
    method,
    pruning_amount,
    cfg,
    log=False, 
    device=None,
):

    def train_with_cfg(model, cfg, opt, return_best, freeze_zero):
        return train(model, small_train_loader, big_test_loader, cfg.epochs, cfg.lr, cfg.lr_decay_milestones, cfg.lr_decay_gamma, 
                     cfg.weight_decay, log=log, return_best=return_best, opt=opt, freeze_zero=freeze_zero)
    def finetune(model, mode):
        if mode == 'after pruning':
            train_acc, train_loss, val_acc, val_loss = train_with_cfg(model, cfg.finetune.after_pruning, cfg.opt, True, freeze_zero=True)
            return train_acc, train_loss, val_acc, val_loss
        elif mode == 'after metanetwork':
            train_acc, train_loss, val_acc, val_loss = train_with_cfg(model, cfg.finetune.after_metanetwork, cfg.opt, False, freeze_zero=False)
            return train_acc, train_loss, val_acc, val_loss
        else:
            raise NotImplementedError

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.eval().to(device)
    if log:
        logger = get_logger("Unstructured Pruning")
    final_model = state_dict_to_model(model_name, origin_state_dict).eval().to(device)
    node_index, node_features, edge_index, edge_features = state_dict_to_graph(model_name, origin_state_dict)
    if metanetwork != -1:
        metanetwork.eval().to(device)
        node_pred, edge_pred = metanetwork.forward(node_features.to(device), edge_index.to(device), edge_features.to(device))
        model = graph_to_model(model_name, origin_state_dict, node_index, node_pred, edge_index, edge_pred)
        train_acc, train_loss, val_acc, val_loss = finetune(model, mode='after metanetwork')
    else:
        node_pred, edge_pred = deepcopy(node_features), deepcopy(edge_features)
        model = graph_to_model(model_name, origin_state_dict, node_index, node_pred, edge_index, edge_pred)
        if log:
            logger.info("No metanetwork, use original node and edge features.")
    
    model, base_params, left_params = unstructured_pruning(model, dataset_name, big_train_loader, big_test_loader, pruning_amount, method, log=log, device=device)
    train_acc, train_loss, val_acc, val_loss = finetune(model, mode='after pruning')
    final_model = state_dict_to_model(model_name, model.state_dict())
    if log:
        logger.info(f"\n{final_model}")
        logger.info(f"Origin val acc : {info['val_acc']} Final val acc : {val_acc}\n\
                      Origin params: {base_params}   Params left: {left_params}   Pruned params ratio:{(base_params - left_params) / base_params:.4f}")
    
    return model