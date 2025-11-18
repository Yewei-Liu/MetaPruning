import torch
import torch.nn as nn
import torch_pruning as tp


@torch.no_grad()
def count_model_flops_and_params(model, example_inputs, verbose=False):
    """
    Wraps torch-pruning's count_ops_and_params so that:
      - dense_flops  == tp.utils.count_ops_and_params(...)
      - sparse_flops scales Conv/Linear layers by weight nnz ratio
      - total_params and total_nnz are computed over all parameters
    """
    # 1. Get dense FLOPs and params from torch-pruning (this uses the opcounter you pasted)
    dense_flops, dense_params, layer_flops, layer_params = tp.utils.count_ops_and_params(
        model,
        example_inputs=example_inputs,
        layer_wise=True,
    )

    # 2. Compute sparse FLOPs:
    #    For Conv/Linear: scale by (nnz_weight / total_weight)
    #    For others (BN, pooling, activations, etc.): keep original FLOPs
    sparse_flops = 0.0
    for m in model.modules():
        lf = layer_flops.get(m)
        if lf is None:
            continue

        if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)) and hasattr(m, "weight"):
            W = m.weight
            total_w = W.numel()
            nnz_w = (W != 0).sum().item()
            ratio = 0.0 if total_w == 0 else nnz_w / total_w
            sparse_flops += lf * ratio
        else:
            # non-pruned / non-weight layers → FLOPs unaffected by sparsity
            sparse_flops += lf
    sparse_flops /= 2.0
    
    # 3. Total params and non-zero params over the whole model
    total_params = 0
    total_nnz = 0
    for p in model.parameters():
        total_params += p.numel()
        total_nnz += (p != 0).sum().item()
    sparsity = 0.0 if total_params == 0 else 1.0 - total_nnz / total_params

    if verbose:
        print("==== FLOPs & Params (torch-pruning consistent) ====")
        print(f"Dense FLOPs / sample : {dense_flops:.3f}")
        print(f"Sparse FLOPs / sample: {sparse_flops:.3f}")
        print(f"Total params         : {total_params}")
        print(f"Total non-zero       : {total_nnz}")
        print(f"Sparsity             : {sparsity:.4f}")
        print("===================================================")

    return dense_flops, sparse_flops, total_params, total_nnz, layer_flops, layer_params
