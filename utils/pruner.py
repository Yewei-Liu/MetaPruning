from functools import partial
import torch
import torch_pruning as tp
from utils.dict import dataset_num_classes_dict

def get_pruner(model, 
               example_inputs,
               reg,
               dataset_name,
               method,
               iterative_steps=200,
               max_pruning_ratio=300,
               global_pruning=True):
    
    # For more infomation, please refer to code "https://github.com/VainF/Torch-Pruning/blob/master/torch_pruning/pruner/importance.py" and their corresponding paper.
    if method == "random":
        imp = tp.importance.RandomImportance()
        pruner_entry = partial(tp.pruner.GroupNormPruner, reg=reg, global_pruning=global_pruning)
    elif method == "group_l2_norm_no_normalizer":
        imp = tp.importance.GroupMagnitudeImportance(p=2, group_reduction='mean', normalizer=None)
        pruner_entry = partial(tp.pruner.GroupNormPruner, reg=reg, global_pruning=global_pruning)
    elif method == "group_l2_norm_mean_normalizer":
        imp = tp.importance.GroupMagnitudeImportance(p=2, group_reduction='mean', normalizer='mean')
        pruner_entry = partial(tp.pruner.GroupNormPruner, reg=reg, global_pruning=global_pruning)
    elif method == "group_l2_norm_max_normalizer":
        imp = tp.importance.GroupMagnitudeImportance(p=2, group_reduction='mean', normalizer='max') 
        pruner_entry = partial(tp.pruner.GroupNormPruner, reg=reg, global_pruning=global_pruning)
    else:
        raise NotImplementedError(f"Method {method} not implemented !")
    
    unwrapped_parameters = []
    ignored_layers = []
    pruning_ratio_dict = {}
    
    # ignore output layers
    try:
        num_classes = dataset_num_classes_dict[dataset_name]
    except KeyError:
        raise KeyError(f"Dataset '{dataset_name}' not found in dataset_num_classes_dict")
    for m in model.modules():
        if isinstance(m, torch.nn.Linear) and m.out_features == num_classes:
            ignored_layers.append(m)
        elif isinstance(m, torch.nn.modules.conv._ConvNd) and m.out_channels == num_classes:
            ignored_layers.append(m)
    
    # Here we fix iterative_steps=200 to prune the model progressively with small steps 
    # until the required speed up is achieved.
    pruner = pruner_entry(
        model,
        example_inputs,
        importance=imp,
        iterative_steps=iterative_steps,
        pruning_ratio=1.0,
        pruning_ratio_dict=pruning_ratio_dict,
        max_pruning_ratio=max_pruning_ratio,
        ignored_layers=ignored_layers,
        unwrapped_parameters=unwrapped_parameters,
    )
    return pruner