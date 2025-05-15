import logging
import numpy as np
from copy import deepcopy
import torch
import hydra
from omegaconf import OmegaConf
import os
import collections

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import destroy_process_group
import torch_geometric
import matplotlib.pyplot as plt
import networkx as nx
import torchvision.transforms as transforms
from torch.func import functional_call
import torchvision
import torch_pruning as tp
from generate_dataset.resnet_deep_family import resnet50

from datasets import Dataset
from utils.logging import get_logger
from utils.visualize import visualize_subgraph
from utils.convert import graph_to_state_dict, state_dict_to_model, state_dict_to_graph, graph_to_model
from generate_dataset.resnet_family import resnet56, MyResNet
from generate_dataset.VGG_family import vgg19_bn, MyVGG
from utils.pruning import get_pruner, adaptive_pruning, pruning_one_step, progressive_pruning
from utils.train import train, eval
from utils.visualize import visualize_acc_speed_up_curve
from utils.meta_train import meta_train, meta_eval
from utils.seed import set_seed

from data_loaders.get_dataset_model import get_dataset_model_loader
from data_loaders.get_dataset import get_dataset_loader


@hydra.main(config_path="configs", config_name="base", version_base=None)
def main(cfg):
    if cfg.run == 'meta_train':
        level = str(cfg.level)
    elif cfg.run == 'visualize':
        if cfg.index == 'train':
            raise ValueError("must use : python main.py run=visualize index=<metanetwork_index> or <metanetwork_index_list>")
        index_list = cfg.index
        if isinstance(index_list, int):
            index_list = [index_list]
        index_list_name = ''.join(str(i)+'_' for i in index_list) 
    elif cfg.run == 'pruning_one_step': # pruning with only one metanetwork
        if cfg.index == 'train':
            raise ValueError("must use : python main.py run=pruning_one_step index=<metanetwork_index>")
        index = cfg.index
    elif cfg.run == 'pruning_final':
        if cfg.index == 'train':
            raise ValueError("must use : python main.py run=pruning_final index=<pruning_index>")
        reproduce_dir = os.path.join('final', cfg.task.task_name, f'reproduce_{cfg.reproduce_index}')
        pruning_index = cfg.index
    elif cfg.run == 'visualize_final':
        if cfg.index == 'train':
            raise ValueError("must use : python main.py run=visualize_final index=<reproduce_index>")
        reproduce_dir = os.path.join('final', cfg.task.task_name, f'reproduce_{cfg.index}')
    elif cfg.run == 'pretrain_final':
        if cfg.index == 'train':
            raise ValueError("must use : python main.py run=pretrain_final index=<reproduce_index>")
        reproduce_dir = os.path.join('final', cfg.task.task_name, f'reproduce_{cfg.index}')
    elif cfg.run == 'test':
        if cfg.index == 'train':
            raise ValueError("must use : python main.py run=pretrain_final index=<reproduce_index>")
        reproduce_dir = os.path.join('final', cfg.task.task_name, f'reproduce_{cfg.index}')
    else:
        raise ValueError(f"run {cfg.run} is not valid")

    logging.info(f'\n\n{OmegaConf.to_yaml(cfg)}')
    log = cfg.log
    run = cfg.run
    seed = cfg.seed
    set_seed(seed)
    
    cfg = cfg.task

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_train_loader, model_val_loader = get_dataset_model_loader(cfg.dataset_model)
    big_train_loader, big_test_loader = get_dataset_loader(cfg.big_batch_dataset)
    small_train_loader, small_test_loader = get_dataset_loader(cfg.small_batch_dataset)

    if run == 'meta_train':
        metanetwork_config = OmegaConf.select(cfg.metanetwork, level)
        metanetwork = hydra.utils.instantiate(metanetwork_config).to(device)
        metanetwork = meta_train(metanetwork, model_train_loader, big_train_loader, small_train_loader, cfg.meta_train, log=log,
                                 model_val_loader=model_val_loader, big_data_val_loader=big_test_loader)
    
    elif run == 'visualize':
        # TO DO: change to a list of index
        save_dir = cfg.meta_train.save_path
        all_files = os.listdir(save_dir)
        model, origin_state_dict, info, node_index, node_features, edge_index, edge_features = next(iter(model_val_loader))
        def load_metanetwork(index):
            prefix = f"epoch_{index}"
            matching_files = [
            os.path.join(save_dir, file)
            for file in all_files
            if (file.startswith(prefix + '.') or file.startswith(prefix + '_')) and os.path.isfile(os.path.join(save_dir, file))
            ]
            if len(matching_files) == 0:
                raise ValueError(f"no metanetwork found with index {index}")
            elif len(matching_files) >= 2:
                raise ValueError(f"More than one metanetwork found with index {index}")
            metanetwork = torch.load(matching_files[0], weights_only=False)
            return metanetwork
        def get_new_model(metanetwork):
            node_pred, edge_pred = metanetwork.forward(node_features.to(device), edge_index.to(device), edge_features.to(device))
            new_model = graph_to_model(cfg.model_name, origin_state_dict, node_index, node_pred, edge_index, edge_pred, device)
            train(new_model, small_train_loader, big_test_loader, cfg.pruning.finetune.after_metanetwork.epochs,
                  cfg.pruning.finetune.after_metanetwork.lr, cfg.pruning.finetune.after_metanetwork.lr_decay_milestones,
                  cfg.pruning.finetune.after_metanetwork.lr_decay_gamma, cfg.pruning.finetune.after_metanetwork.weight_decay,
                  log=log, return_best=True, opt=cfg.pruning.opt)
            return new_model
        model_list = [model]
        label_list = ['origin']
        for index in index_list:
            metanetwork = load_metanetwork(index)
            new_model = get_new_model(metanetwork)
            model_list.append(new_model)
            label_list.append(f'epoch_{index}')
        visualize_acc_speed_up_curve( model_list, cfg.dataset.dataset_name, label_list,
                                      big_test_loader, [info['current_speed_up'] for i in range(len(model_list))], cfg.visualize.max_speed_up, cfg.meta_train.method,
                                      cfg.visualize.marker, save_dir=cfg.visualize.save_path, name=f"{index_list_name}.png",
                                      ylim=cfg.visualize.ylim, log=log, figsize=cfg.visualize.figsize, font_scale=cfg.visualize.font_scale)
    
    elif run == 'pruning_one_step':
        save_dir = cfg.meta_train.save_path
        all_files = os.listdir(save_dir)
        prefix = f"epoch_{index}"
        matching_files = [
        os.path.join(save_dir, file)
        for file in all_files
        if (file.startswith(prefix + '.') or file.startswith(prefix + '_')) and os.path.isfile(os.path.join(save_dir, file))
        ]
        if len(matching_files) == 0:
            raise ValueError(f"no metanetwork found with index {index}")
        elif len(matching_files) >= 2:
            raise ValueError(f"More than one metanetwork found with index {index}")
        metanetwork = torch.load(matching_files[0], weights_only=False)
        model, origin_state_dict, info, node_index, node_features, edge_index, edge_features = next(iter(model_val_loader))
        pruning_one_step(model, cfg.model_name, cfg.dataset.dataset_name, info, origin_state_dict, big_train_loader, small_train_loader, 
                         big_test_loader, metanetwork, cfg.pruning, info['current_speed_up'], log=log)

    elif run == 'pruning_final':
        model = torch.load(os.path.join(reproduce_dir, 'model.pth'))
        if isinstance(model, collections.OrderedDict):
            model = state_dict_to_model(cfg.model_name, model)
        model.to(device)
        train_acc, train_loss = eval(model, big_train_loader)
        val_acc, val_loss = eval(model, big_test_loader)
        current_speed_up = 1.0
        info = {'train_acc': train_acc, 'train_loss': train_loss, 'val_acc': val_acc, 'val_loss': val_loss, 'current_speed_up': current_speed_up}
        logging.info(f"Before pruning:\n{info}")
        cfg.pruning.pruning_index = pruning_index
        if cfg.task_name == 'resnet56_on_CIFAR10':
            metanetwork = torch.load(os.path.join(reproduce_dir, 'metanetwork.pth'))
            speed_up, model = progressive_pruning(model, cfg.dataset.dataset_name, big_train_loader, 
                                                big_test_loader, 1.32, log=log)
            current_speed_up *= speed_up
            train(model, small_train_loader, big_test_loader, 80, 0.01, "40, 70", log=log)
            model = pruning_one_step(model, cfg.model_name, cfg.dataset.dataset_name, info, model.state_dict(), 
                                               big_train_loader, small_train_loader, big_test_loader, metanetwork,
                                               cfg.pruning, current_speed_up, log=log)
        elif cfg.task_name == 'VGG19_on_CIFAR100':
            metanetwork = torch.load(os.path.join(reproduce_dir, 'metanetwork.pth'))
            speed_up, model = progressive_pruning(model, cfg.dataset.dataset_name, big_train_loader, 
                                                big_test_loader, 2.0, log=log)
            current_speed_up *= speed_up
            train(model, small_train_loader, big_test_loader, 140, 0.01, "80, 120", log=log) 
            model = pruning_one_step(model, cfg.model_name, cfg.dataset.dataset_name, info, model.state_dict(), 
                                               big_train_loader, small_train_loader, big_test_loader, metanetwork,
                                               cfg.pruning, current_speed_up, log=log)
            
    elif run == 'visualize_final':
        model = torch.load(os.path.join(reproduce_dir, 'model.pth'))
        model_list = [state_dict_to_model(cfg.model_name, model.state_dict())]
        model.to(device)
        train_acc, train_loss = eval(model, big_train_loader)
        val_acc, val_loss = eval(model, big_test_loader)
        current_speed_up = 1.0
        info = {'train_acc': train_acc, 'train_loss': train_loss, 'val_acc': val_acc, 'val_loss': val_loss, 'current_speed_up': current_speed_up}
        logging.info(f"Before pruning:\n{info}")
        if cfg.task_name == 'resnet56_on_CIFAR10':
            metanetwork = torch.load(os.path.join(reproduce_dir, 'metanetwork.pth'))
            metanetwork.eval().to(device)
            speed_up, model = progressive_pruning(model, cfg.dataset.dataset_name, big_train_loader, 
                                                big_test_loader, 1.32, log=log)
            current_speed_up *= speed_up
            train(model, small_train_loader, big_test_loader, 80, 0.01, "40, 70", log=log)
            node_index, node_features, edge_index, edge_features = state_dict_to_graph(cfg.model_name, model.state_dict())
            node_pred, edge_pred = metanetwork.forward(node_features.to(device), edge_index.to(device), edge_features.to(device))
            model = graph_to_model(cfg.model_name, model.state_dict(), node_index, node_pred, edge_index, edge_pred, device)
            train(model, small_train_loader, big_test_loader, cfg.pruning.finetune.after_metanetwork.epochs,
                  cfg.pruning.finetune.after_metanetwork.lr, cfg.pruning.finetune.after_metanetwork.lr_decay_milestones,
                  cfg.pruning.finetune.after_metanetwork.lr_decay_gamma, cfg.pruning.finetune.after_metanetwork.weight_decay,
                  log=log, return_best=True)
            model_list.append(model)
            label_list = ['origin', 'pruned']
            base_speed_up_list = [1.0, current_speed_up]
            visualize_acc_speed_up_curve(   model_list, cfg.dataset.dataset_name, label_list,
                                            big_test_loader, base_speed_up_list, cfg.visualize.max_speed_up, cfg.meta_train.method,
                                            cfg.visualize.marker, save_dir=reproduce_dir, name="visualize.png",
                                            ylim=cfg.visualize.ylim, log=log)
    
    elif run == 'pretrain_final':
        if cfg.task_name == 'resnet56_on_CIFAR10':
            model = resnet56(10)
            train(model, small_train_loader, big_test_loader, 200, 0.1, "100, 150, 180", log=log)
            torch.save(model, os.path.join(reproduce_dir, 'model.pth'))
        elif cfg.task_name == 'VGG19_on_CIFAR100':
            model = vgg19_bn(num_classes=100)
            train(model, small_train_loader, big_test_loader, 200, 0.1, "100,150,180", log=log)
            torch.save(model, os.path.join(reproduce_dir, 'model.pth'))
        else:
            raise ValueError(f"task {cfg.task_name} is not valid")
            
    elif run == 'test':
        model = resnet50(1000)
        model_name = 'resnet50'
        node_index, node_features, edge_index, edge_features = state_dict_to_graph(model_name, model.state_dict())
        new_model = graph_to_model(model_name, model.state_dict(), node_index, node_features, edge_index, edge_features, device)
        error = 0
        for key in new_model.state_dict().keys():
            error = ((model.state_dict()[key] - new_model.state_dict()[key])**2).sum()
            print(key, error)
      

if __name__ == "__main__":
    main()