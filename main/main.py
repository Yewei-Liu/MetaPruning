import logging
import numpy as np
from copy import deepcopy
import torch
import hydra
from omegaconf import OmegaConf
import os
import collections
from utils.convert import graph_to_state_dict, state_dict_to_model, state_dict_to_graph, graph_to_model
from generate_dataset.resnet_family import resnet56, resnet110
from generate_dataset.resnet_deep_family import myresnet18, myresnet26
from generate_dataset.VGG_family import vgg19_bn
from utils.pruning import get_pruner, adaptive_pruning, pruning_one_step, progressive_pruning, unstructured_pruning_one_step
from utils.train import train, eval
from utils.visualize import visualize_acc_speed_up_curve, visualize_acc_pruned_params_curve
from utils.meta_train import meta_train, meta_eval
from utils.seed import set_seed
from utils.analyse import (
    analyze_models, 
    plot_activation_and_grad, 
    plot_bn_stats, plot_conv_weight_norms, 
    plot_global_taylor_and_ratio, heatmap_metric, 
    scatter_corr_vs_norm, 
    plot_conv_norms_by_layer, 
    plot_conv_norm_ratio,
    compare_models_layer_hist,
    plot_inter_channel_corr_by_layer,
    plot_effective_rank_by_layer,
    compare_taylor_sensitivity_hist,
    plot_taylor_sensitivity_by_layer
    )
from data_loaders.get_dataset_model import get_dataset_model_loader
from data_loaders.get_dataset import get_dataset_loader

@hydra.main(config_path="configs", config_name="base", version_base=None)
def main(cfg):
    if cfg.run == 'meta_train': 
        pass
    elif cfg.run == 'visualize':
        if cfg.index == 'train':
            raise ValueError("must use : python main.py run=visualize index=<metanetwork_index> or <metanetwork_index_list> (index_list starts with 0 means visualize origin network without metanetwork, otherwise not)")
        index_list = cfg.index
        if isinstance(index_list, int):
            index_list = [index_list]
        index_list_name = ''.join(str(i)+'_' for i in index_list) 
        visualize_origin = False
        if index_list[0] == 0:
            index_list = index_list[1:]
            visualize_origin = True
    elif cfg.run == 'analyse':
        if cfg.index == 'train':
            raise ValueError("must use : python main.py run=analyse index=<metanetwork_index> or <metanetwork_index_list> (index_list starts with 0 means analyse origin network without metanetwork, otherwise not)")
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
        reproduce_dir = os.path.join('final', cfg.task.task_name, f'{cfg.method}', f'reproduce_{cfg.reproduce_index}')
        pruning_index = cfg.index
    elif cfg.run == 'unstructured_pruning_final':
        assert cfg.method.startswith('unstructured'), f"only unstructured pruning is supported in unstructured_pruning_final, method now {cfg.method}"
        if cfg.index == 'train':
            raise ValueError("must use : python main.py run=pruning_final index=pruing_amount")
        reproduce_dir = os.path.join('final', cfg.task.task_name, f'{cfg.method}', f'reproduce_{cfg.reproduce_index}')
        pruning_amount = cfg.index
    elif cfg.run == 'visualize_final':
        if cfg.index == 'train':
            raise ValueError("must use : python main.py run=visualize_final index=<reproduce_index>")
        reproduce_dir = os.path.join('final', cfg.task.task_name, f'{cfg.method}', f'reproduce_{cfg.index}')
    elif cfg.run == 'pretrain_final':
        if cfg.index == 'train':
            raise ValueError("must use : python main.py run=pretrain_final index=<reproduce_index>")
        reproduce_dir = os.path.join('final', cfg.task.task_name, f'{cfg.method}', f'reproduce_{cfg.index}')
        os.makedirs(reproduce_dir, exist_ok=True)
    elif cfg.run == 'test':
        if cfg.index == 'train':
            raise ValueError("must use : python main.py run=pretrain_final index=<reproduce_index>")
        reproduce_dir = os.path.join('final', cfg.task.task_name, f'{cfg.method}', f'reproduce_{cfg.index}')
    else:
        raise ValueError(f"run {cfg.run} is not valid")

    logging.info(f'\n\n{OmegaConf.to_yaml(cfg)}')
    log = cfg.log
    run = cfg.run
    seed = cfg.seed
    method = cfg.method
    set_seed(seed)
    
    cfg = cfg.task

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_train_loader = None
    try:
        model_train_loader, model_val_loader = get_dataset_model_loader(cfg.dataset_model)
    except:
        print('No dataset models exist, just for quick reproduce.')
    big_train_loader, big_test_loader = get_dataset_loader(cfg.big_batch_dataset)
    small_train_loader, small_test_loader = get_dataset_loader(cfg.small_batch_dataset)
    
    if run == 'meta_train':
        metanetwork = hydra.utils.instantiate(cfg.metanetwork).to(device)
        if model_train_loader is None:
            raise ValueError('Dataset models doesn\'t exist !')
        metanetwork = meta_train(metanetwork, model_train_loader, big_train_loader, small_train_loader, cfg.meta_train, log=log,
                                 model_val_loader=model_val_loader, big_data_val_loader=big_test_loader)
    
    elif run == 'visualize':
        resume_path = os.path.join(cfg.visualize.save_path, f"{os.path.splitext(index_list_name)[0]}.pkl")
        if not os.path.exists(resume_path):
            resume_path = None
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
            freeze_zero = True if method.startswith('unstructured') else False
            if resume_path is not None:
                return model
            node_pred, edge_pred = metanetwork.forward(node_features.to(device), 
                                                       [ei.to(device) for ei in edge_index] if isinstance(edge_index, list) else edge_index.to(device), 
                                                       [ef.to(device) for ef in edge_features] if isinstance(edge_features, list) else edge_features.to(device))
            new_model = graph_to_model(cfg.model_name, origin_state_dict, node_index, node_pred, edge_index, edge_pred, device)
            train(new_model, small_train_loader, big_test_loader, cfg.pruning.finetune.after_metanetwork.epochs,
                  cfg.pruning.finetune.after_metanetwork.lr, cfg.pruning.finetune.after_metanetwork.lr_decay_milestones,
                  cfg.pruning.finetune.after_metanetwork.lr_decay_gamma, cfg.pruning.finetune.after_metanetwork.weight_decay,
                  log=log, return_best=True, opt=cfg.pruning.opt, freeze_zero=freeze_zero)
            return new_model
        model_list = []
        label_list = []
        if visualize_origin is True:
            model_list = [model]
            label_list = ['origin']

        for index in index_list:
            metanetwork = load_metanetwork(index)
            new_model = get_new_model(metanetwork)
            model_list.append(new_model)
            label_list.append(f'epoch_{index}')
        if method.startswith('unstructured'):
            visualize_acc_pruned_params_curve( model_list, cfg.dataset.dataset_name, label_list,
                            big_test_loader, 0.95, method,
                            cfg.visualize.marker, save_dir=cfg.visualize.save_path, name=f"{index_list_name}.png",
                            ylim=cfg.visualize.ylim, log=log, figsize=cfg.visualize.figsize, font_scale=cfg.visualize.font_scale, resume_path=resume_path)
        else:
            visualize_acc_speed_up_curve( model_list, cfg.dataset.dataset_name, label_list,
                            big_test_loader, [info['current_speed_up'] for i in range(len(model_list))], cfg.visualize.max_speed_up, method,
                            cfg.visualize.marker, save_dir=cfg.visualize.save_path, name=f"{index_list_name}.png",
                            ylim=cfg.visualize.ylim, log=log, figsize=cfg.visualize.figsize, font_scale=cfg.visualize.font_scale, resume_path=resume_path)
    
    elif run == 'analyse':
        resume_path = os.path.join(cfg.analyse.save_path, f"{index_list_name}.pth")
        resume_ckpt = None
        if os.path.exists(resume_path):
            resume_ckpt = torch.load(resume_path, weights_only=False)
        else:
            resume_path = None
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
            freeze_zero = True if method.startswith('unstructured') else False
            node_pred, edge_pred = metanetwork.forward(node_features.to(device), 
                                                       [ei.to(device) for ei in edge_index] if isinstance(edge_index, list) else edge_index.to(device), 
                                                       [ef.to(device) for ef in edge_features] if isinstance(edge_features, list) else edge_features.to(device))
            new_model = graph_to_model(cfg.model_name, origin_state_dict, node_index, node_pred, edge_index, edge_pred, device)
            train(new_model, small_train_loader, big_test_loader, cfg.pruning.finetune.after_metanetwork.epochs,
                  cfg.pruning.finetune.after_metanetwork.lr, cfg.pruning.finetune.after_metanetwork.lr_decay_milestones,
                  cfg.pruning.finetune.after_metanetwork.lr_decay_gamma, cfg.pruning.finetune.after_metanetwork.weight_decay,
                  log=log, return_best=True, opt=cfg.pruning.opt, freeze_zero=freeze_zero)
            return new_model

        model_list = [model if resume_ckpt is None else resume_ckpt['origin_model']]
        new_resume_ckpt = {'origin_model': model_list[0]}
        label_list = ['origin']

        for index in index_list:
            if resume_ckpt is None:
                metanetwork = load_metanetwork(index)
                new_model = get_new_model(metanetwork)
                model_list.append(new_model)
                label_list.append(f'epoch_{index}')
                new_resume_ckpt[f'epoch_{index}_model'] = new_model
            else:
                model_list.append(resume_ckpt[f'epoch_{index}_model'])
                label_list.append(f'epoch_{index}')
        if resume_path is None:
            os.makedirs(os.path.join(cfg.analyse.save_path), exist_ok=True)
            torch.save(new_resume_ckpt, os.path.join(cfg.analyse.save_path, f"{index_list_name}.pth"))
        
        res = analyze_models(model_list, big_test_loader, device, num_batches=10, bn_gamma_zero_threshold=0.001, entropy_bins=40)
        print(res)
        os.makedirs(os.path.join(cfg.analyse.save_path, index_list_name), exist_ok=True)
        # plot_conv_norms_by_layer(res, [0, 1], 'weight_l2', os.path.join(cfg.analyse.save_path, index_list_name))
        # plot_conv_norms_by_layer(res, [0, 1], 'weight_l1', os.path.join(cfg.analyse.save_path, index_list_name))
        # plot_conv_norm_ratio(res, 1, 0, 'weight_l2', os.path.join(cfg.analyse.save_path, index_list_name))
        # plot_conv_norm_ratio(res, 1, 0, 'weight_l1', os.path.join(cfg.analyse.save_path, index_list_name))
        
        # compare_models_layer_hist(
        #     model_list,
        #     0,
        #     1,
        #     os.path.join(cfg.analyse.save_path, index_list_name, "norm"),
        #     "conv",
        #     bins=20,
        #     xlim=(0.0, 1.0)
        # )
        
        # plot_inter_channel_corr_by_layer(
        #     res,
        #     [0, 1],
        #     save_path=os.path.join(cfg.analyse.save_path, index_list_name, "corr")
        # )
        
        # plot_effective_rank_by_layer(
        #     res,
        #     [0, 1],
        #     save_path=os.path.join(cfg.analyse.save_path, index_list_name, "erank")
        # )
        
        compare_taylor_sensitivity_hist(
            model_list,
            0,
            1,
            os.path.join(cfg.analyse.save_path, index_list_name, "taylor_sensitivity"),
            big_test_loader,
            device,
            num_batches=10,
            layer_kinds="conv",
            bins=40,
            xlim=(0.0, 0.004)
        )
        
        # plot_taylor_sensitivity_by_layer(
        #     res,
        #     [0,1],
        #     save_path=os.path.join(cfg.analyse.save_path, index_list_name, "taylor_sensitivity_line_chart")
        # )

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
        model = torch.load(os.path.join(reproduce_dir, 'model.pth'), weights_only=False)
        if isinstance(model, collections.OrderedDict):
            model = state_dict_to_model(cfg.model_name, model)
        model.to(device)
        train_acc, train_loss = eval(model, big_train_loader)
        val_acc, val_loss = eval(model, big_test_loader)
        current_speed_up = 1.0
        info = {'train_acc': train_acc, 'train_loss': train_loss, 'val_acc': val_acc, 'val_loss': val_loss, 'current_speed_up': current_speed_up}
        logging.info(f"Before pruning:\n{info}")
        cfg.pruning.pruning_index = pruning_index
        file_path = os.path.join(reproduce_dir, 'metanetwork.pth')
        if os.path.isfile(file_path):
            metanetwork = torch.load(file_path, weights_only=False)
        else:
            print(f"Error: File '{file_path}' not found. Don't use metanetwork.")
            metanetwork = -1

        if cfg.task_name in ['resnet56_on_CIFAR10', 'resnet56_on_CIFAR100', 'resnet56_on_SVHN', 'resnet110_on_CIFAR10']:
            speed_up, model = progressive_pruning(model, cfg.dataset.dataset_name, big_train_loader, 
                                                big_test_loader, cfg.initial_pruning_speed_up, method=method, log=log)
            current_speed_up *= speed_up
            train(model, small_train_loader, big_test_loader, 80, 0.01, "40, 70", log=log)
        elif cfg.task_name == 'VGG19_on_CIFAR100':
            speed_up, model = progressive_pruning(model, cfg.dataset.dataset_name, big_train_loader, 
                                                big_test_loader, cfg.initial_pruning_speed_up, method=method, log=log)
            current_speed_up *= speed_up
            train(model, small_train_loader, big_test_loader, 140, 0.01, "80, 120", log=log) 
        elif cfg.task_name in ['resnet56(no_init_pruning)_on_CIFAR10']:
            print("No init pruning.")
        else:
            raise NotImplementedError(f"task {cfg.task_name} is not supported for final pruning")
        
        model = pruning_one_step(model, cfg.model_name, cfg.dataset.dataset_name, info, model.state_dict(), 
                                            big_train_loader, small_train_loader, big_test_loader, metanetwork,
                                            cfg.pruning, current_speed_up, log=log)
    
    elif run == 'unstructured_pruning_final':
        model = torch.load(os.path.join(reproduce_dir, 'model.pth'), weights_only=False)
        if isinstance(model, collections.OrderedDict):
            model = state_dict_to_model(cfg.model_name, model)
        model.to(device)
        train_acc, train_loss = eval(model, big_train_loader)
        val_acc, val_loss = eval(model, big_test_loader)
        info = {'train_acc': train_acc, 'train_loss': train_loss, 'val_acc': val_acc, 'val_loss': val_loss}
        logging.info(f"Before pruning:\n{info}")
        file_path = os.path.join(reproduce_dir, 'metanetwork.pth')
        if os.path.isfile(file_path):
            metanetwork = torch.load(file_path, weights_only=False)
        else:
            print(f"Error: File '{file_path}' not found. Don't use metanetwork.")
            metanetwork = -1
        model = unstructured_pruning_one_step(model, cfg.model_name, cfg.dataset.dataset_name, info, model.state_dict(), 
                                            big_train_loader, small_train_loader, big_test_loader, metanetwork,
                                            method, pruning_amount, cfg.pruning, log, device)

        
        
            
    elif run == 'visualize_final':
        model = torch.load(os.path.join(reproduce_dir, 'model.pth'), weights_only=False)
        model.to(device)
        train_acc, train_loss = eval(model, big_train_loader)
        val_acc, val_loss = eval(model, big_test_loader)
        current_speed_up = 1.0
        info = {'train_acc': train_acc, 'train_loss': train_loss, 'val_acc': val_acc, 'val_loss': val_loss, 'current_speed_up': current_speed_up}
        logging.info(f"Before pruning:\n{info}")
        metanetwork = torch.load(os.path.join(reproduce_dir, 'metanetwork.pth'), weights_only=False)
        metanetwork.eval().to(device)
        if cfg.task_name in ['resnet56_on_CIFAR10', 'resnet56_on_CIFAR100', 'resnet56_on_SVHN', 'resnet110_on_CIFAR10']:
            speed_up, model = progressive_pruning(model, cfg.dataset.dataset_name, big_train_loader, 
                                                big_test_loader, cfg.initial_pruning_speed_up, method=method, log=log)
            current_speed_up *= speed_up
            train(model, small_train_loader, big_test_loader, 80, 0.01, "40, 70", log=log)
        elif cfg.task_name == 'VGG19_on_CIFAR100':
            speed_up, model = progressive_pruning(model, cfg.dataset.dataset_name, big_train_loader, 
                                                big_test_loader, cfg.initial_pruning_speed_up, method=method, log=log)
            current_speed_up *= speed_up
            train(model, small_train_loader, big_test_loader, 140, 0.01, "80, 120", log=log)
        else:
            raise NotImplementedError(f"task {cfg.task_name} is not supported for final visualization")
        model_list = [state_dict_to_model(cfg.model_name, model.state_dict())]
        node_index, node_features, edge_index, edge_features = state_dict_to_graph(cfg.model_name, model.state_dict())
        node_pred, edge_pred = metanetwork.forward(node_features.to(device), 
                                                   [ei.to(device) for ei in edge_index] if isinstance(edge_index, list) else edge_index.to(device), 
                                                   [ef.to(device) for ef in edge_features] if isinstance(edge_features, list) else edge_features.to(device))
        model = graph_to_model(cfg.model_name, model.state_dict(), node_index, node_pred, edge_index, edge_pred, device)
        train(model, small_train_loader, big_test_loader, cfg.pruning.finetune.after_metanetwork.epochs,
                cfg.pruning.finetune.after_metanetwork.lr, cfg.pruning.finetune.after_metanetwork.lr_decay_milestones,
                cfg.pruning.finetune.after_metanetwork.lr_decay_gamma, cfg.pruning.finetune.after_metanetwork.weight_decay,
                log=log, return_best=True)
        model_list.append(model)
        label_list = ['origin', 'metanetwork']
        base_speed_up_list = [current_speed_up, current_speed_up]
        visualize_acc_speed_up_curve(   model_list, cfg.dataset.dataset_name, label_list,
                                        big_test_loader, base_speed_up_list, cfg.visualize.max_speed_up, method,
                                        cfg.visualize.marker, save_dir=reproduce_dir, name="visualize.png",
                                        ylim=cfg.visualize.ylim, log=log)
    
    elif run == 'pretrain_final':
        if cfg.task_name == 'resnet56_on_CIFAR10':
            model = resnet56(10)
            train(model, small_train_loader, big_test_loader, 200, 0.1, "100, 150, 180", log=log)
            torch.save(model, os.path.join(reproduce_dir, 'model.pth'))
        elif cfg.task_name == 'resnet56_on_CIFAR100':
            model = resnet56(100)
            train(model, small_train_loader, big_test_loader, 200, 0.1, "100, 150, 180", log=log)
            torch.save(model, os.path.join(reproduce_dir, 'model.pth'))
        elif cfg.task_name == 'resnet56_on_SVHN':
            model = resnet56(10)
            train(model, small_train_loader, big_test_loader, 200, 0.1, "100, 150, 180", log=log)
            torch.save(model, os.path.join(reproduce_dir, 'model.pth'))
        elif cfg.task_name == 'resnet110_on_CIFAR10':
            model = resnet110(10)
            train(model, small_train_loader, big_test_loader, 200, 0.1, "100, 150, 180", log=log)
            torch.save(model, os.path.join(reproduce_dir, 'model.pth'))
        elif cfg.task_name == 'VGG19_on_CIFAR100':
            model = vgg19_bn(num_classes=100)
            train(model, small_train_loader, big_test_loader, 200, 0.1, "100, 150, 180", log=log)
            torch.save(model, os.path.join(reproduce_dir, 'model.pth'))
        elif cfg.task_name == 'resnet56(no_init_pruning)_on_CIFAR10':
            model = resnet56(10)
            train(model, small_train_loader, big_test_loader, 200, 0.1, "100, 150, 180", log=log)
            torch.save(model, os.path.join(reproduce_dir, 'model.pth'))
        else:
            raise NotImplementedError(f"task {cfg.task_name} is not supported for pretrain final")
            
    elif run == 'test':
        pass

if __name__ == "__main__":
    main()