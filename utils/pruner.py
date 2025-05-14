from functools import partial
import torch
import torch_pruning as tp


dataset_num_classes_dict = {
    'CIFAR10': 10,
    'CIFAR100': 100,
}


def get_pruner(model, 
               example_inputs,
               reg,
               dataset_name,
               iterative_steps=200,
               max_pruning_ratio=300,
               method = "group_sl",
               global_pruning=True):
    if method == "group_norm":
        imp = tp.importance.GroupMagnitudeImportance(p=2)
        pruner_entry = partial(tp.pruner.GroupNormPruner, reg=reg, global_pruning=global_pruning)
    elif method == "group_sl":
        imp = tp.importance.GroupMagnitudeImportance(p=2, normalizer='max') # normalized by the maximum score for CIFAR
        pruner_entry = partial(tp.pruner.GroupNormPruner, reg=reg, global_pruning=global_pruning)
    else:
        raise NotImplementedError(f"Method {method} is not a valid method.")
    #args.is_accum_importance = is_accum_importance
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