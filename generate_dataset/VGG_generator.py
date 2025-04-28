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
            node_index, node_features, edge_index, edge_features = state_dict_to_graph(cfg.model_name, model.state_dict())
            node_pred, edge_pred = metanetwork.forward(node_features.to(device), edge_index.to(device), edge_features.to(device))
            model = graph_to_model(cfg.model_name, model.state_dict(), node_index, node_pred, edge_index, edge_pred)
            train_acc, train_loss, val_acc, val_loss = train(model, small_train_loader, big_test_loader, 100, 0.01, "60, 80", return_best=True, log=log)
            speed_up, model = progressive_pruning(model, cfg.dataset_name, big_train_loader, big_test_loader, cfg.speed_up[j], method=method, log=log)
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
    index = 1
    dataset_path = cfg.dataset_path
    num = cfg.num
    train_loader, test_loader = get_dataset_loader(cfg.dataset)



    # node_num = [3, 64, 64, 128, 128, 256, 256, 256, 256, 512, 512, 512, 512, 512, 512, 512, 512, 100]
    # # node_num = [3, 20, 64, 128, 128, 256, 256, 256, 256, 150, 50, 10, 10, 20, 100, 100, 200, 100]
    level=1
    metanetwork_dir = os.path.join('metanetwork', f'{cfg.model_name}_on_{cfg.dataset_name}')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    current_speed_up = 1.0
    model = vgg19_bn(num_classes=100)
    train_acc, train_loss, val_acc, val_loss = train(model, train_loader, test_loader, 300, 0.1, "150, 250", log=True)
    torch.save(model, "after_train.pth")
    for j in range(level):
        metanetwork = torch.load(os.path.join(metanetwork_dir, f'{j}.pth'))
        metanetwork.to(device)
        node_index, node_features, edge_index, edge_features = state_dict_to_graph(cfg.model_name, model.state_dict())
        node_pred, edge_pred = metanetwork.forward(node_features.to(device), edge_index.to(device), edge_features.to(device))
        model = graph_to_model(cfg.model_name, model.state_dict(), node_index, node_pred, edge_index, edge_pred)
        train_acc, train_loss, val_acc, val_loss = train(model, train_loader, test_loader, 300, 0.01, "150, 250", return_best=True, log=True)
        torch.save(model, f"after_metanetwork_{j}.pth")
        speed_up, model = progressive_pruning(model, cfg.dataset_name, train_loader, test_loader, 2.2, method=cfg.method)
        current_speed_up *= speed_up
        train_acc, train_loss, val_acc, val_loss = train(model, train_loader, test_loader, 300, 0.01, "150, 250", return_best=True, log=True) 
        torch.save(model, f"after_pruning_{j}.pth")


if __name__ == "__main__":
    main()