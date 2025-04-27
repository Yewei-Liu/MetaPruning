import torch
from generate_dataset.resnet_deep_family import resnet50
import hydra
from omegaconf import OmegaConf
import time
import logging
from utils.pruning import adaptive_pruning, progressive_pruning
from utils.train import train, eval
from data_loaders.get_dataset import get_dataset_loader
from utils.visualize import visualize_acc_speed_up_curve
from utils.convert import state_dict_to_graph, graph_to_model
import os



def resnet56_on_CIFAR10_generator(num, level, method, cfg, cfg_big_dataset, cfg_small_dataset):
    big_train_loader, big_test_loader = get_dataset_loader(cfg_big_dataset)
    small_train_loader, small_test_loader = get_dataset_loader(cfg_small_dataset)
    acc_threshold = cfg.acc_threshold
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    metanetwork_dir = os.path.join('metanetwork', f'{cfg.model_name}_on_{cfg.dataset_name}')
    log = True
    def one_generate_step():
        model = resnet50(1000)
        train_acc, train_loss, val_acc, val_loss = train(model, small_train_loader, big_test_loader, 200, 0.1, "100,150,180", log=log)
        current_speed_up, model = progressive_pruning(model, cfg.dataset_name, big_train_loader, big_test_loader, 1.32, cfg.method, log=log)
        train_acc, train_loss, val_acc, val_loss = train(model, small_train_loader, big_test_loader, 80, 0.01, "40, 70", log=log) 
        if level == 0:
            data = model.eval().cpu().state_dict()
            data['train_acc'] = train_acc
            data['train_loss'] = train_loss
            data['val_acc'] = val_acc
            data['val_loss'] = val_loss        
            data['current_speed_up'] = current_speed_up
            yield data
        for j in range(level):
            metanetwork = torch.load(os.path.join(metanetwork_dir, f'{j}.pth'))
            metanetwork.to(device)
            node_index, node_features, edge_index, edge_features = state_dict_to_graph(cfg.model_name, model.cpu().state_dict())
            node_pred, edge_pred = metanetwork.forward(node_features.to(device), edge_index.to(device), edge_features.to(device))
            model = graph_to_model(cfg.model_name, model.state_dict(), node_index, node_pred, edge_index, edge_pred)
            train_acc, train_loss, val_acc, val_loss = train(model, small_train_loader, big_test_loader, 100, 0.01, "60, 80", return_best=True, log=log)
            speed_up, model = progressive_pruning(model, cfg.dataset_name, big_train_loader, big_test_loader, cfg.speed_up, cfg.method, log=log)
            current_speed_up *= speed_up
            train_acc, train_loss, val_acc, val_loss = train(model, small_train_loader, big_test_loader, 140, 0.01, "80, 120", return_best=True, log=log) 
            if j == level - 1:
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
    level = cfg.level
    cfg_dataset = cfg.dataset
    cfg = cfg.data_generator.cfg
    train_loader, test_loader = get_dataset_loader(cfg_dataset)
    acc_threshold = cfg.acc_threshold
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    metanetwork_dir = os.path.join('metanetwork', f'{cfg.model_name}_on_{cfg.dataset_name}')
    # model = resnet56(10)
    # train_acc, train_loss, val_acc, val_loss = train(model, train_loader, test_loader, 200, 0.1, "100,150,180", log=True)
    # torch.save(model, 'after_train.pth')
    # current_speed_up, model = adaptive_pruning(model, cfg.model_name, cfg.dataset_name, train_loader, test_loader, acc_threshold, log=True)
    # train_acc, train_loss, val_acc, val_loss = train(model, train_loader, test_loader, 80, 0.01, "40, 70", log=True) 
    # torch.save(model, 'after_pruning.pth')
    model = torch.load('after_pruning.pth')
    for j in range(level):
        metanetwork = torch.load(os.path.join(metanetwork_dir, f'{j}.pth'))
        metanetwork.to(device)
        node_index, node_features, edge_index, edge_features = state_dict_to_graph(cfg.model_name, model.cpu().state_dict())
        node_pred, edge_pred = metanetwork.forward(node_features.to(device), edge_index.to(device), edge_features.to(device))
        model = graph_to_model(cfg.model_name, model.state_dict(), node_index, node_pred, edge_index, edge_pred)   
        train_acc, train_loss, val_acc, val_loss = train(model, train_loader, test_loader, 100, 0.01, "60, 80", return_best=True, log=True)
        torch.save(model, f'after_metanetwork_{j}.pth')
        speed_up, model = progressive_pruning(model, cfg.dataset_name, train_loader, test_loader, 2.2, log=True)
        # speed_up, model = adaptive_pruning(model, cfg.model_name, cfg.dataset_name, train_loader, test_loader, acc_threshold, log=True)
        # current_speed_up *= speed_up
        train_acc, train_loss, val_acc, val_loss = train(model, train_loader, test_loader, 100, 0.01, "60, 80", return_best=True, log=True) 
        torch.save(model, f'after_pruning_{j}.pth')



if __name__ == "__main__":
    main()