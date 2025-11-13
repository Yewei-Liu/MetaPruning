import torch
from generate_dataset.resnet_family import resnet56, resnet110
from generate_dataset.resnet_deep_family import myresnet26, myresnet18
import hydra
from omegaconf import OmegaConf
import time
import logging
from utils.pruning import adaptive_pruning, progressive_pruning
from utils.train import train, eval
from data_loaders.get_dataset import get_dataset_loader
from utils.visualize import visualize_acc_speed_up_curve
from utils.convert import state_dict_to_graph, graph_to_model
from utils.dict import dataset_num_classes_dict
import os



def resnet56_on_CIFAR10_generator(num, cfg, cfg_big_dataset, cfg_small_dataset):
    big_train_loader, big_test_loader = get_dataset_loader(cfg_big_dataset)
    small_train_loader, small_test_loader = get_dataset_loader(cfg_small_dataset)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    metanetwork_dir = os.path.join('metanetwork', f'{cfg.model_name}_on_{cfg.dataset_name}')
    log = True
    def one_generate_step():
        model = resnet56(dataset_num_classes_dict[cfg.dataset_name])
        train_acc, train_loss, val_acc, val_loss = train(model, small_train_loader, big_test_loader, 200, 0.1, "100,150,180", log=log)
        current_speed_up, model = progressive_pruning(model, cfg.dataset_name, big_train_loader, big_test_loader, cfg.initial_speed_up, cfg.method, log=log)
        train_acc, train_loss, val_acc, val_loss = train(model, small_train_loader, big_test_loader, 80, 0.01, "40, 70", log=log) 
        data = model.eval().cpu().state_dict()
        data['train_acc'] = train_acc
        data['train_loss'] = train_loss
        data['val_acc'] = val_acc
        data['val_loss'] = val_loss        
        data['current_speed_up'] = current_speed_up
        yield data
    for i in range(num):
        yield from one_generate_step()
        
def resnet56_on_CIFAR10_no_init_pruning_generator(num, cfg, cfg_big_dataset, cfg_small_dataset):
    big_train_loader, big_test_loader = get_dataset_loader(cfg_big_dataset)
    small_train_loader, small_test_loader = get_dataset_loader(cfg_small_dataset)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    metanetwork_dir = os.path.join('metanetwork', f'{cfg.model_name}_on_{cfg.dataset_name}')
    log = True
    def one_generate_step():
        model = resnet56(dataset_num_classes_dict[cfg.dataset_name])
        train_acc, train_loss, val_acc, val_loss = train(model, small_train_loader, big_test_loader, 200, 0.1, "100,150,180", log=log)
        current_speed_up = 1.0
        data = model.eval().cpu().state_dict()
        data['train_acc'] = train_acc
        data['train_loss'] = train_loss
        data['val_acc'] = val_acc
        data['val_loss'] = val_loss        
        data['current_speed_up'] = current_speed_up
        yield data
    for i in range(num):
        yield from one_generate_step()

def resnet110_on_CIFAR10_generator(num, cfg, cfg_big_dataset, cfg_small_dataset):
    big_train_loader, big_test_loader = get_dataset_loader(cfg_big_dataset)
    small_train_loader, small_test_loader = get_dataset_loader(cfg_small_dataset)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    metanetwork_dir = os.path.join('metanetwork', f'{cfg.model_name}_on_{cfg.dataset_name}')
    log = True
    def one_generate_step():
        model = resnet110(dataset_num_classes_dict[cfg.dataset_name])
        train_acc, train_loss, val_acc, val_loss = train(model, small_train_loader, big_test_loader, 200, 0.1, "100,150,180", log=log)
        current_speed_up, model = progressive_pruning(model, cfg.dataset_name, big_train_loader, big_test_loader, cfg.initial_speed_up, cfg.method, log=log)
        train_acc, train_loss, val_acc, val_loss = train(model, small_train_loader, big_test_loader, 80, 0.01, "40, 70", log=log) 
        data = model.eval().cpu().state_dict()
        data['train_acc'] = train_acc
        data['train_loss'] = train_loss
        data['val_acc'] = val_acc
        data['val_loss'] = val_loss        
        data['current_speed_up'] = current_speed_up
        yield data
    for i in range(num):
        yield from one_generate_step()

def resnet18_on_CIFAR10_generator(num, cfg, cfg_big_dataset, cfg_small_dataset):
    big_train_loader, big_test_loader = get_dataset_loader(cfg_big_dataset)
    small_train_loader, small_test_loader = get_dataset_loader(cfg_small_dataset)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    metanetwork_dir = os.path.join('metanetwork', f'{cfg.model_name}_on_{cfg.dataset_name}')
    log = True
    def one_generate_step():
        model = myresnet18(dataset_num_classes_dict[cfg.dataset_name])
        train_acc, train_loss, val_acc, val_loss = train(model, small_train_loader, big_test_loader, 200, 0.1, "100,150,180", log=log)
        current_speed_up, model = progressive_pruning(model, cfg.dataset_name, big_train_loader, big_test_loader, cfg.initial_speed_up, cfg.method, log=log)
        train_acc, train_loss, val_acc, val_loss = train(model, small_train_loader, big_test_loader, 80, 0.01, "40, 70", log=log) 
        data = model.eval().cpu().state_dict()
        data['train_acc'] = train_acc
        data['train_loss'] = train_loss
        data['val_acc'] = val_acc
        data['val_loss'] = val_loss        
        data['current_speed_up'] = current_speed_up
        yield data
    for i in range(num):
        yield from one_generate_step()

def resnet26_on_CIFAR10_generator(num, cfg, cfg_big_dataset, cfg_small_dataset):
    big_train_loader, big_test_loader = get_dataset_loader(cfg_big_dataset)
    small_train_loader, small_test_loader = get_dataset_loader(cfg_small_dataset)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    metanetwork_dir = os.path.join('metanetwork', f'{cfg.model_name}_on_{cfg.dataset_name}')
    log = True
    def one_generate_step():
        model = myresnet26(dataset_num_classes_dict[cfg.dataset_name])
        train_acc, train_loss, val_acc, val_loss = train(model, small_train_loader, big_test_loader, 200, 0.1, "100,150,180", log=log)
        current_speed_up, model = progressive_pruning(model, cfg.dataset_name, big_train_loader, big_test_loader, cfg.initial_speed_up, cfg.method, log=log)
        train_acc, train_loss, val_acc, val_loss = train(model, small_train_loader, big_test_loader, 80, 0.01, "40, 70", log=log) 
        data = model.eval().cpu().state_dict()
        data['train_acc'] = train_acc
        data['train_loss'] = train_loss
        data['val_acc'] = val_acc
        data['val_loss'] = val_loss        
        data['current_speed_up'] = current_speed_up
        yield data
    for i in range(num):
        yield from one_generate_step()

@hydra.main(config_path="configs", config_name="base", version_base=None)
def main(cfg):
    cfg_dataset = cfg.small_batch_dataset
    cfg = cfg.data_generator.cfg
    train_loader, test_loader = get_dataset_loader(cfg_dataset)
    model = myresnet26(dataset_num_classes_dict[cfg_dataset.dataset_name])
    train_acc, train_loss, val_acc, val_loss = train(model, train_loader, test_loader, 200, 0.1, "100,150,180", log=True)
    visualize_acc_speed_up_curve(model, cfg.dataset_name, 'origin', test_loader, 1.0, 10.0, cfg.method, save_dir='./', name=f'{cfg.model_name}.png')


if __name__ == "__main__":
    main()