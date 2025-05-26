from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_pruning as tp
from copy import deepcopy
from utils.logging import get_logger
from utils.convert import state_dict_to_model, state_dict_to_graph, graph_to_state_dict, graph_to_model
from utils.train import train, eval
from utils.pruner import get_pruner


def progressive_pruning(
        model, 
        dataset_name,
        train_loader,
        test_loader,
        speed_up, 
        method = 'group_sl',
        log=False, 
        device=None,
        eval_train_data=True,
        eval_test_data=True,
):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.eval().to(device)
    if dataset_name.lower() in ['cifar10', 'cifar100']:
        example_inputs = torch.ones((1, 3, 32, 32)).to(device)
    elif dataset_name.lower() in ['imagenet']:
        example_inputs = torch.ones((1, 3, 224, 224)).to(device)
    else:
        raise NotImplementedError(f"Dataset {dataset_name} is not supported.")
    pruner = get_pruner(model, example_inputs, 0.1, dataset_name, method=method)
    base_ops, _ = tp.utils.count_ops_and_params(model, example_inputs=example_inputs)
    current_speed_up = 1.0
    if log:
        logger = get_logger("Progressive pruning")
    while current_speed_up < speed_up :
        pruner.step()
        pruned_ops, _ = tp.utils.count_ops_and_params(model, example_inputs=example_inputs)
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
    return current_speed_up, model


def adaptive_pruning(
        
        model, 
        model_name,
        dataset_name,
        train_loader, 
        test_loader, 
        acc_threshold=0.9, 
        method = 'group_sl',
        log=False, 
        device=None,
        target_speed_up=float('inf')):
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
                     metanetwork, 
                     cfg,
                     current_speed_up=1.0, 
                     log=False, 
                     device=None,
):
    # Pruning_index < 1.0 : adaptive_pruning. Otherwise >= 1.0 : progressive_pruning.
    if isinstance(cfg.pruning_index, float):
        pruning_index = cfg.pruning_index 
    else:
        pruning_index = cfg.pruning_index[cfg.level]
    method = cfg.method
    # adaptive_pruning_first = cfg.adaptive_pruning_first

    def train_with_cfg(model, cfg, opt):
        return train(model, small_train_loader, big_test_loader, cfg.epochs, cfg.lr, cfg.lr_decay_milestones, cfg.lr_decay_gamma, 
                     cfg.weight_decay, log=log, return_best=True, opt=opt)
    def finetune(model, mode):
        if mode == 'after pruning':
            train_acc, train_loss, val_acc, val_loss = train_with_cfg(model, cfg.finetune.after_pruning, cfg.opt)
            return train_acc, train_loss, val_acc, val_loss
        elif mode == 'after metanetwork':
            train_acc, train_loss, val_acc, val_loss = train_with_cfg(model, cfg.finetune.after_metanetwork, cfg.opt)
            return train_acc, train_loss, val_acc, val_loss
        else:
            raise NotImplementedError

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.eval().to(device)
    metanetwork.eval().to(device)
    if log:
        logger = get_logger("Pruning")
    final_model = state_dict_to_model(model_name, origin_state_dict).eval().to(device)
    node_index, node_features, edge_index, edge_features = state_dict_to_graph(model_name, origin_state_dict)
    node_pred, edge_pred = metanetwork.forward(node_features.to(device), edge_index.to(device), edge_features.to(device))
    model = graph_to_model(model_name, origin_state_dict, node_index, node_pred, edge_index, edge_pred)
    train_acc, train_loss, val_acc, val_loss = finetune(model, mode='after metanetwork')
    if pruning_index < 1.0:
        speed_up, model = adaptive_pruning(model, model_name, dataset_name, big_train_loader, big_test_loader, pruning_index, method, log=log)
    else:
        assert pruning_index > current_speed_up, f"pruning_index {pruning_index} should be larger than current speed up {current_speed_up}"
        speed_up, model = progressive_pruning(model, dataset_name, big_train_loader, big_test_loader, pruning_index / current_speed_up, method=method, log=log)
    train_acc, train_loss, val_acc, val_loss = finetune(model, mode='after pruning')
    final_model = state_dict_to_model(model_name, model.state_dict())
    current_speed_up *= speed_up
    if log:
        logger.info(f"\n{final_model}")
        logger.info(f"Origin val acc : {info['val_acc']} Final val acc : {val_acc}\n\
                      Speed up: {speed_up:.2f}   Final speed up: {current_speed_up:.2f}")

    return final_model



# def iterative_pruning(model, 
#                       model_name,
#                       dataset_name, 
#                       info,
#                       origin_state_dict,
#                       train_loader, 
#                       test_loader, 
#                       metanetwork, 
#                       cfg,
#                       current_speed_up=1.0, 
#                       log=False, 
#                       device=None,
#                       target_speed_up=float('inf'),
#                       max_iter=100000,
#                       ):
#     # Pruning_index < 1.0 : adaptive_pruning. Otherwise >= 1.0 : progressive_pruning.
#     pruning_index = cfg.pruning_index 
#     finetuning_acc_threshold = cfg.finetuning_acc_threshold
#     method = cfg.method
#     begin_fast_mode_iter = cfg.begin_fast_mode_iter
#     adaptive_pruning_first = cfg.adaptive_pruning_first

#     def train_with_cfg(model, cfg):
#         return train(model, train_loader, test_loader, cfg.epochs, cfg.lr, cfg.lr_decay_milestones, cfg.lr_decay_gamma, 
#                      cfg.weight_decay, log=log, return_best=True)
#     def finetune(model, mode):
#         if mode == 'after pruning':
#             train_acc, train_loss, val_acc, val_loss = train_with_cfg(model, cfg.finetune.after_pruning)
#             return train_acc, train_loss, val_acc, val_loss
#         elif mode == 'after metanetwork':
#             train_acc, train_loss, val_acc, val_loss = train_with_cfg(model, cfg.finetune.after_metanetwork)
#             return train_acc, train_loss, val_acc, val_loss
#         elif mode == 'fast after metanetwork':
#             train_acc, train_loss, val_acc, val_loss = train_with_cfg(model, cfg.finetune.fast_after_metanetwork)
#             return train_acc, train_loss, val_acc, val_loss
#         elif mode == 'final':
#             if cfg.finetune.final.epochs > 0:
#                 train_acc, train_loss, val_acc, val_loss = train_with_cfg(model, cfg.finetune.final)
#             else:
#                 train_acc, train_loss = eval(model, train_loader)
#                 val_acc, val_loss = eval(model, test_loader)
#             return train_acc, train_loss, val_acc, val_loss
#         else:
#             raise NotImplementedError

#     if device is None:
#         device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     model.eval().to(device)
#     metanetwork.eval().to(device)
#     if log:
#         logger = get_logger("Iterative pruning")
#     final_model = state_dict_to_model(model_name, origin_state_dict).eval().to(device)
#     iter = 0
#     while iter < max_iter:
#         iter += 1
#         if iter >= 2 or not adaptive_pruning_first:
#             node_index, node_features, edge_index, edge_features = state_dict_to_graph(model_name, origin_state_dict)
#             node_pred, edge_pred = metanetwork.forward(node_features.to(device), edge_index.to(device), edge_features.to(device))
#             model = graph_to_model(model_name, origin_state_dict, node_index, node_pred, edge_index, edge_pred)
#             if iter >= begin_fast_mode_iter:
#                 train_acc, train_loss, val_acc, val_loss = finetune(model, mode='fast after metanetwork')
#             else:
#                 train_acc, train_loss, val_acc, val_loss = finetune(model, mode='after metanetwork')
#             if val_acc < finetuning_acc_threshold:
#                 break
#         if pruning_index < 1.0:
#             speed_up, model = adaptive_pruning(model, model_name, dataset_name, train_loader, test_loader, pruning_index, method, log=log, target_speed_up=target_speed_up)
#         else:
#             speed_up, model = progressive_pruning(model, dataset_name, train_loader, test_loader, pruning_index, method=method, log=log)
#         if speed_up <= 1.01:
#             break
#         train_acc, train_loss, val_acc, val_loss = finetune(model, mode='after pruning')
#         if val_acc < finetuning_acc_threshold:
#             break
#         final_model = state_dict_to_model(model_name, model.state_dict())
#         current_speed_up *= speed_up
#         origin_state_dict = deepcopy(model.cpu().state_dict())
#         if log:
#             logger.info(f"\n{final_model}")
#             logger.info(f"Current speed up: {current_speed_up:.2f}")
#     final_train_acc, final_train_loss, final_val_acc, final_val_loss = finetune(final_model, mode='final')
#     if log:
#         logger.info(f"Origin val acc : {info['val_acc']} Final val acc : {final_val_acc}   Final speed up: {current_speed_up:.2f}")

#     return final_model


