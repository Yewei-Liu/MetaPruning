import torch
from torch import nn
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.utils import to_networkx
from torch_geometric.data import Data
import numpy as np
from utils.pruning import get_pruner
import torch_pruning as tp
from utils.mylogging import get_logger
from utils.train import eval
import os
import pickle

def get_acc_speed_up_list(
        model,
        dataset_name, 
        test_loader,
        base_speed_up,
        max_speed_up = 5.0,
        method = 'group_sl'
):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.eval().to(device)
    example_inputs = torch.ones((1, 3, 32, 32)).to(device)
    pruner = get_pruner(model, example_inputs, 0.1, dataset_name, method=method)
    base_ops, _ = tp.utils.count_ops_and_params(model, example_inputs=example_inputs)
    current_speed_up = 1.0
    val_acc, val_loss = eval(model, test_loader, device)
    acc_list = [val_acc]
    speed_up_list = [base_speed_up]
    while current_speed_up < max_speed_up / base_speed_up:
        pruner.step()
        pruned_ops, _ = tp.utils.count_ops_and_params(model, example_inputs=example_inputs)
        val_acc, val_loss = eval(model, test_loader, device)
        current_speed_up = float(base_ops) / pruned_ops
        acc_list.append(val_acc)
        speed_up_list.append(current_speed_up * base_speed_up)
    del pruner
    return acc_list, speed_up_list

def visualize_acc_speed_up_curve(
        models,
        dataset_name,
        labels,
        test_loader,
        base_speed_up,
        max_speed_up=5.0,
        method='group_sl', 
        marker='o',
        save_dir='tmp/',
        name='tmp.png',
        ylim=(0.0, 1.0),
        log=True,
        figsize=(20, 20),
        font_scale=1.5,  # New parameter to control font scaling
        resume_path=None,
):
    os.makedirs(save_dir, exist_ok=True)
    if log:
        logger = get_logger("Visualize acc speed up curve")
        logger.info("Start visualizing")
    
    # Set larger font sizes
    plt.rcParams.update({
        'font.size': 12 * font_scale,           # General font size
        'axes.titlesize': 16 * font_scale,      # Title font size
        'axes.labelsize': 14 * font_scale,      # X and Y labels font size
        'xtick.labelsize': 12 * font_scale,     # X-axis tick labels
        'ytick.labelsize': 12 * font_scale,      # Y-axis tick labels
        'legend.fontsize': 12 * font_scale,      # Legend font size
        'figure.titlesize': 18 * font_scale      # Figure title size
    })
    
    plt.figure(figsize=figsize)
    
    if resume_path is not None:
        if log:
            logger.info(f"Load existing statistics from {resume_path}, no need to recalculate.")
        with open(resume_path, "rb") as f:
            t = pickle.load(f)
            acc_list_dict = t['acc_list_dict']
            speed_up_list_dict = t['speed_up_list_dict']
    else:
        acc_list_dict = {}
        speed_up_list_dict = {}
            
    
    if isinstance(models, list):
        assert isinstance(base_speed_up, list), 'if models are list, base_speed_up must be list !'
        for i, m in enumerate(models):
            if resume_path is not None:
                acc_list, speed_up_list = acc_list_dict[i], speed_up_list_dict[i]
            else:
                acc_list, speed_up_list = get_acc_speed_up_list(m, dataset_name, test_loader, base_speed_up[i], max_speed_up, method)
                acc_list_dict[i], speed_up_list_dict[i] = acc_list, speed_up_list
            plt.plot(speed_up_list, acc_list, marker=marker, label=labels[i], markersize=4*font_scale, linewidth=2*font_scale)
            if log:
                logger.info(f"Model {i+1}/{len(models)} visualized")
    else:
        if resume_path is not None:
            acc_list, speed_up_list = acc_list_dict[0], speed_up_list_dict[0]
        else:
            acc_list, speed_up_list = get_acc_speed_up_list(models, dataset_name, test_loader, base_speed_up, max_speed_up, method)
            acc_list_dict[0], speed_up_list_dict[0] = acc_list, speed_up_list 
        plt.plot(speed_up_list, acc_list, marker=marker, label=labels, markersize=4*font_scale, linewidth=2*font_scale)
    
    plt.xlabel('Speed Up', fontsize=14 * font_scale)  # You can override individual elements if needed
    plt.ylabel('Test Acc', fontsize=14 * font_scale)
    plt.title('Test Acc vs. Speed Up', fontsize=16 * font_scale)
    plt.xlim(1.0, max_speed_up)
    plt.ylim(ylim)
    plt.locator_params(axis='y', nbins=20)
    plt.grid()
    
    # Make legend larger
    plt.legend(loc='upper right', prop={'size': 12 * font_scale})
    
    # Adjust tick label size
    plt.tick_params(axis='both', which='major', labelsize=12 * font_scale)
    
    plt.savefig(os.path.join(save_dir, name), dpi=300, bbox_inches='tight')  # Higher DPI and tight layout
    plt.close()  # Close the figure to free memory
    
    if resume_path is None:
        save_path = os.path.join(save_dir, f"{os.path.splitext(name)[0]}.pkl")
        with open(save_path, 'wb') as f:
            pickle.dump({'acc_list_dict': acc_list_dict, 'speed_up_list_dict': speed_up_list_dict}, f)
        if log:
            logger.info(f"Statistics saved to {save_path}")
    
    if log:
        for i in range(len(acc_list_dict)):
            logger.info({f"{speed_up_list_dict[i][j]:.2f}": f"{acc_list_dict[i][j]:.4f}" for j in range(len(acc_list_dict[i]))})
        logger.info("End visualizing")
    return acc_list_dict, speed_up_list_dict

