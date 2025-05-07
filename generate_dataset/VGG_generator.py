import torch
from generate_dataset.VGG_family import vgg19_bn, MyVGG
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


def VGG19_on_CIFAR100_generator(level, method, num, cfg, cfg_big_dataset, cfg_small_dataset):
    big_train_loader, big_test_loader = get_dataset_loader(cfg_big_dataset)
    small_train_loader, small_test_loader = get_dataset_loader(cfg_small_dataset)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    metanetwork_dir = os.path.join('metanetwork', f'{cfg.model_name}_on_{cfg.dataset_name}')
    log = True
    def one_generate_step():
        current_speed_up = 1.0
        model = vgg19_bn(num_classes=100)
        train_acc, train_loss, val_acc, val_loss = train(model, small_train_loader, big_test_loader, 200, 0.1, "100,150,180", log=log)
        current_speed_up, model = progressive_pruning(model, cfg.dataset_name, big_train_loader, big_test_loader, 2.0, cfg.method, log=log)
        train_acc, train_loss, val_acc, val_loss = train(model, small_train_loader, big_test_loader, 140, 0.01, "80, 120", log=log) 
        if level == 0:
            data = model.eval().cpu().state_dict()
            data['train_acc'] = train_acc
            data['train_loss'] = train_loss
            data['val_acc'] = val_acc
            data['val_loss'] = val_loss        
            data['current_speed_up'] = current_speed_up
            yield data
        for j in range(level):
            def _one_iter():
                nonlocal model, current_speed_up
                metanetwork = torch.load(os.path.join(metanetwork_dir, f'{j}.pth'))
                metanetwork.to(device)
                node_index, node_features, edge_index, edge_features = state_dict_to_graph(cfg.model_name, model.state_dict())
                node_pred, edge_pred = metanetwork.forward(node_features.to(device), edge_index.to(device), edge_features.to(device))
                model = graph_to_model(cfg.model_name, model.state_dict(), node_index, node_pred, edge_index, edge_pred)
                train_acc, train_loss, val_acc, val_loss = train(model, small_train_loader, big_test_loader, 100, 0.01, "60, 80", return_best=True, log=log)
                speed_up, model = progressive_pruning(model, cfg.dataset_name, big_train_loader, big_test_loader, cfg.speed_up[j] / current_speed_up, method=method, log=log)
                current_speed_up *= speed_up
                train_acc, train_loss, val_acc, val_loss = train(model, small_train_loader, big_test_loader, 140, 0.01, "80, 120", return_best=True, log=log) 
                logging.info(f'level : {j + 1}, train acc: {train_acc}, train loss: {train_loss}, val acc: {val_acc}, val loss: {val_loss}')
                if j == level - 1:
                    data = model.eval().cpu().state_dict()
                    data['train_acc'] = train_acc
                    data['train_loss'] = train_loss
                    data['val_acc'] = val_acc
                    data['val_loss'] = val_loss        
                    data['current_speed_up'] = current_speed_up
                    yield data
            yield from _one_iter()
    for i in range(num):
        yield from one_generate_step()
        


@hydra.main(config_path="configs", config_name="base", version_base=None)
def main(cfg):
    big_train_loader, big_test_loader = get_dataset_loader(cfg.big_batch_dataset)
    small_train_loader, small_test_loader = get_dataset_loader(cfg.small_batch_dataset)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    metanetwork_dir = os.path.join('metanetwork', f'{cfg.model_name}_on_{cfg.dataset_name}')
    current_speed_up = 1.0
    model = vgg19_bn(num_classes=100)
    log=True
    # train_acc, train_loss, val_acc, val_loss = train(model, small_train_loader, big_test_loader, 200, 0.1, "100,150,180", log=log)
    # torch.save(model, "after_train.pth")
    model = torch.load("after_train.pth")
    speed_up, model = progressive_pruning(model, cfg.dataset_name, big_train_loader, big_test_loader, 2.0, cfg.method, log=log)
    current_speed_up *= speed_up
    train_acc, train_loss, val_acc, val_loss = train(model, small_train_loader, big_test_loader, 140, 0.01, "80, 120", log=log) 
    torch.save(model, "after_pruning_2.0.pth")
    speed_up, model = progressive_pruning(model, cfg.dataset_name, big_train_loader, big_test_loader, 4.0 / current_speed_up, cfg.method, log=log)
    current_speed_up *= speed_up
    train_acc, train_loss, val_acc, val_loss = train(model, small_train_loader, big_test_loader, 140, 0.01, "80, 120", log=log) 
    torch.save(model, "after_pruning_4.0.pth")


    # level=1
    # metanetwork_dir = os.path.join('metanetwork', f'{cfg.model_name}_on_{cfg.dataset_name}')
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # current_speed_up = 1.0
    # model = vgg19_bn(num_classes=100)
    # train_acc, train_loss, val_acc, val_loss = train(model, train_loader, test_loader, 300, 0.1, "150, 250", log=True)
    # torch.save(model, "after_train.pth")
    # for j in range(level):
    #     metanetwork = torch.load(os.path.join(metanetwork_dir, f'{j}.pth'))
    #     metanetwork.to(device)
    #     node_index, node_features, edge_index, edge_features = state_dict_to_graph(cfg.model_name, model.state_dict())
    #     node_pred, edge_pred = metanetwork.forward(node_features.to(device), edge_index.to(device), edge_features.to(device))
    #     model = graph_to_model(cfg.model_name, model.state_dict(), node_index, node_pred, edge_index, edge_pred)
    #     train_acc, train_loss, val_acc, val_loss = train(model, train_loader, test_loader, 300, 0.01, "150, 250", return_best=True, log=True)
    #     torch.save(model, f"after_metanetwork_{j}.pth")
    #     speed_up, model = progressive_pruning(model, cfg.dataset_name, train_loader, test_loader, 2.2, method=cfg.method)
    #     current_speed_up *= speed_up
    #     train_acc, train_loss, val_acc, val_loss = train(model, train_loader, test_loader, 300, 0.01, "150, 250", return_best=True, log=True) 
    #     torch.save(model, f"after_pruning_{j}.pth")


if __name__ == "__main__":
    main()