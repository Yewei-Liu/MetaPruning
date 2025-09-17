from functools import partial
import torch
import torch_pruning as tp
from utils.dict import dataset_num_classes_dict

def _unwrap_model(model):
    """Return the real model if wrapped by DP/DDP; otherwise the model itself."""
    # Avoid importing DP/DDP types if not available; use duck-typing first.
    if hasattr(model, "module"):
        try:
            from torch.nn.parallel import DataParallel, DistributedDataParallel
            if isinstance(model, (DataParallel, DistributedDataParallel)):
                return model.module
        except Exception:
            # Fallback: many wrappers expose .module; use it if present.
            return model.module
    return model

def get_pruner(model, 
               example_inputs,
               reg,
               dataset_name,
               method,
               iterative_steps=200,
               max_pruning_ratio=300,
               global_pruning=True,
               special_type=None,
               ):
    # --- unwrap parallel wrappers (DP/DDP) before touching attributes ---
    base_model = _unwrap_model(model)

    # --- importance & pruner entry selection ---
    unwrapped_parameters = None
    if special_type == 'vit':
        # Guard against missing attributes in some ViT implementations
        if not hasattr(base_model, "encoder"):
            raise AttributeError("Expected ViT-like model to have 'encoder' attribute.")
        if not hasattr(base_model, "class_token"):
            raise AttributeError("Expected ViT-like model to have 'class_token' attribute.")
        unwrapped_parameters = [
            (base_model.encoder.pos_embedding, 2),
            (base_model.class_token, 2),
        ]

    # For more information, please refer to Torch-Pruning docs/paper.
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
    
    # --- ignore output layers ---
    ignored_layers = []
    pruning_ratio_dict = {}
    try:
        num_classes = dataset_num_classes_dict[dataset_name]
    except KeyError:
        raise KeyError(f"Dataset '{dataset_name}' not found in dataset_num_classes_dict")

    for m in base_model.modules():
        if isinstance(m, torch.nn.Linear) and m.out_features == num_classes:
            ignored_layers.append(m)
        elif isinstance(m, torch.nn.modules.conv._ConvNd) and m.out_channels == num_classes:
            ignored_layers.append(m)
    
    # --- ViT rounding (e.g., keep heads divisible) ---
    round_to = None
    if special_type == 'vit':
        round_to = model.encoder.layers[0].num_heads
    
    # --- build pruner on the unwrapped model ---
    pruner = pruner_entry(
        base_model,                 # IMPORTANT: pass the real model, not DP/DDP wrapper
        example_inputs,
        importance=imp,
        iterative_steps=iterative_steps,
        pruning_ratio=1.0,
        pruning_ratio_dict=pruning_ratio_dict,
        max_pruning_ratio=max_pruning_ratio,
        ignored_layers=ignored_layers,
        round_to=round_to,
        unwrapped_parameters=unwrapped_parameters,
    )
    return pruner

# def get_pruner(model, 
#                example_inputs,
#                reg,
#                dataset_name,
#                method,
#                iterative_steps=200,
#                max_pruning_ratio=300,
#                global_pruning=True,
#                special_type=None,
#                ):
#     unwrapped_parameters = None
#     if special_type == 'vit':
#         unwrapped_parameters = [(model.encoder.pos_embedding, 2), (model.class_token, 2)]
#     # For more infomation, please refer to code "https://github.com/VainF/Torch-Pruning/blob/master/torch_pruning/pruner/importance.py" and their corresponding paper.
#     if method == "random":
#         imp = tp.importance.RandomImportance()
#         pruner_entry = partial(tp.pruner.GroupNormPruner, reg=reg, global_pruning=global_pruning)
#     elif method == "group_l2_norm_no_normalizer":
#         imp = tp.importance.GroupMagnitudeImportance(p=2, group_reduction='mean', normalizer=None)
#         pruner_entry = partial(tp.pruner.GroupNormPruner, reg=reg, global_pruning=global_pruning)
#     elif method == "group_l2_norm_mean_normalizer":
#         imp = tp.importance.GroupMagnitudeImportance(p=2, group_reduction='mean', normalizer='mean')
#         pruner_entry = partial(tp.pruner.GroupNormPruner, reg=reg, global_pruning=global_pruning)
#     elif method == "group_l2_norm_max_normalizer":
#         imp = tp.importance.GroupMagnitudeImportance(p=2, group_reduction='mean', normalizer='max') 
#         pruner_entry = partial(tp.pruner.GroupNormPruner, reg=reg, global_pruning=global_pruning)
#     else:
#         raise NotImplementedError(f"Method {method} not implemented !")
    
#     ignored_layers = []
#     pruning_ratio_dict = {}
    
#     # ignore output layers
#     try:
#         num_classes = dataset_num_classes_dict[dataset_name]
#     except KeyError:
#         raise KeyError(f"Dataset '{dataset_name}' not found in dataset_num_classes_dict")
#     for m in model.modules():
#         if isinstance(m, torch.nn.Linear) and m.out_features == num_classes:
#             ignored_layers.append(m)
#         elif isinstance(m, torch.nn.modules.conv._ConvNd) and m.out_channels == num_classes:
#             ignored_layers.append(m)
    
#     round_to = None
#     if special_type == 'vit':
#         round_to = model.encoder.layers[0].num_heads
    
#     # Here we fix iterative_steps=200 to prune the model progressively with small steps 
#     # until the required speed up is achieved.
#     pruner = pruner_entry(
#         model,
#         example_inputs,
#         importance=imp,
#         iterative_steps=iterative_steps,
#         pruning_ratio=1.0,
#         pruning_ratio_dict=pruning_ratio_dict,
#         max_pruning_ratio=max_pruning_ratio,
#         ignored_layers=ignored_layers,
#         round_to=round_to,
#         unwrapped_parameters=unwrapped_parameters,
#     )
#     return pruner