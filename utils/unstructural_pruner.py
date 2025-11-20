import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from abc import ABC, abstractmethod
from utils.unstructural_flops import count_model_flops_and_params
import torch_pruning as tp


# =========================
# Base pruning method interface
# =========================
class BasePruneMethod(ABC):
    @abstractmethod
    def step(self):
        """Apply one pruning step to the model in-place."""
        pass


# =========================
# Pruner with step() and regularize()
# =========================
class Pruner:
    def __init__(
        self,
        model: nn.Module,
        method_str: str,
        reg_lambda: float = 0.0,
        reg_param_keywords=("weight",),     # only regularize params whose name contains these
        pruning_interval = 0.05,
    ):
        self.model = model
        self.pruning_interval = pruning_interval
        if method_str == "unstructured_l1_norm":
            self.method = UnstructuredMagnitudePrune(module_types=(nn.Linear, nn.Conv2d))
            self.reg_type = "l1"
        elif method_str == "unstructured_l2_norm":
            self.method = UnstructuredMagnitudePrune(module_types=(nn.Linear, nn.Conv2d))
            self.reg_type = "l2"
        elif method_str == "nmsparsity":
            self.method = NMSparsityPrune(module_types=(nn.Linear, nn.Conv2d))
            self.reg_type = None
        else:
            raise NotImplementedError(f"Pruning method '{method_str}' not implemented.")
        
        self.n_steps = 0
        self.reg_lambda = reg_lambda
        self.reg_param_keywords = tuple(reg_param_keywords)

    def step(self, amount=None):
        """Apply pruning once using the underlying method."""
        self.n_steps += 1
        print(self.n_steps)
        if amount is None:
            self.method.step(self.model, amount=min(self.pruning_interval * self.n_steps, 1.0))
        else:
            self.method.step(self.model, amount=amount)
        finalize_pruning(self.model, module_types=(nn.Linear, nn.Conv2d))
        
    def regularize(self, model, alpha=None, bias=None):
        """
        Modify gradients of *prunable* model parameters in-place according to L1 or L2 norm.

        Only regularizes the prunable parts, i.e. the weights of modules that the
        pruning method is configured to prune (e.g. Conv / Linear weights).

        Call this AFTER loss.backward() and BEFORE optimizer.step().
        """
        if self.reg_type is None or self.reg_lambda == 0.0:
            return

        # Try to get the set of module types that are actually pruned
        prunable_module_types = getattr(self.method, "module_types", None)

        # Iterate over modules, not all named_parameters()
        for m in self.model.modules():
            if prunable_module_types is not None and not isinstance(m, prunable_module_types):
                continue

            # For pruned modules, the learnable param is usually:
            #   before pruning: m.weight
            #   after pruning:  m.weight_orig  (with m.weight as a computed tensor)
            for param_attr_name in ("weight_orig", "weight"):
                if not hasattr(m, param_attr_name):
                    continue

                p = getattr(m, param_attr_name)

                # skip if no grad or not trainable
                if not isinstance(p, torch.nn.Parameter):
                    continue
                if not p.requires_grad or p.grad is None:
                    continue

                # optional keyword filter (default: contains "weight")
                if not any(kw in param_attr_name for kw in self.reg_param_keywords):
                    continue

                if self.reg_type == "l2":
                    # grad += λ * w
                    p.grad.add_(self.reg_lambda * p.data)
                elif self.reg_type == "l1":
                    # grad += λ * sign(w)
                    p.grad.add_(self.reg_lambda * p.data.sign())


# =========================
# Unstructured magnitude pruning
# =========================
class UnstructuredMagnitudePrune(BasePruneMethod):
    def __init__(self, module_types=(nn.Linear, nn.Conv2d)):
        """
        amount: fraction of weights (0–1) to prune each step (globally).
        module_types: which module types to prune.
        """
        self.module_types = module_types
    
    def step(self, model: nn.Module, amount: float):
        # collect all (module, 'weight') pairs to prune
        params_to_prune = []
        for m in model.modules():
            if isinstance(m, self.module_types) and hasattr(m, "weight"):
                params_to_prune.append((m, "weight"))

        if not params_to_prune:
            return

        prune.global_unstructured(
            params_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=amount,
        )


# =========================
# N:M sparsity helper
# =========================
def nm_mask(weight: torch.Tensor, N: int, M: int, dim: int = 1) -> torch.Tensor:
    """
    Build an N:M mask along `dim` for `weight`.

    If the size along `dim` is not divisible by M, return an all-ones mask
    (i.e., skip N:M pruning for this layer).

    Args:
        weight: weight tensor
        N, M: N:M sparsity pattern
        dim: dimension along which we apply grouping
    """
    assert N <= M, f"N must be <= M, got N={N}, M={M}"

    # Size along the pruning dimension
    target_dim = weight.size(dim)

    # If dimension is not divisible by M, skip pruning
    if target_dim % M != 0:
        # Return an all-ones mask (no pruning)
        return torch.ones_like(weight, dtype=torch.bool)

    # Safe to prune
    # Move the target dim to the last axis for easier grouping
    w = weight.transpose(dim, -1)
    orig_shape = w.shape
    last_dim = orig_shape[-1]

    num_groups = last_dim // M
    # reshape to (..., num_groups, M)
    w_grouped = w.reshape(*orig_shape[:-1], num_groups, M)
    abs_w = w_grouped.abs()

    # Find top-N by magnitude inside each group of M
    _, topk_idx = torch.topk(abs_w, k=N, dim=-1, largest=True, sorted=False)

    # Build mask with ones in top-N positions
    mask_grouped = torch.zeros_like(w_grouped, dtype=torch.bool)
    mask_grouped.scatter_(-1, topk_idx, True)

    # Reshape mask back to original shape and return
    mask = mask_grouped.reshape(*orig_shape).transpose(dim, -1)
    return mask


# =========================
# N:M sparsity pruning method
#   (supports both Linear and Conv2d)
# =========================
class NMSparsityPrune(BasePruneMethod):
    def __init__(self, module_types=(nn.Linear, nn.Conv2d)):
        """
        Enforce N:M sparsity on `weight` of given module types.
        - N, M: pattern (e.g. 2, 4 for 2:4).
        - dim: which dimension to group on for Linear (usually 1 = in_features).
               For Conv2d we ignore dim and group on flattened kernel.
        """
        self.dim = 1
        self.module_types = module_types

    @torch.no_grad()
    def step(self, model: nn.Module, amount: str):
        N = int(amount.split(":")[0])
        M = int(amount.split(":")[1])
        # Apply N:M pruning to each module
        for m in model.modules():
            if not isinstance(m, self.module_types) or not hasattr(m, "weight"):
                continue

            W = m.weight.data

            # Linear: weight shape (out_features, in_features)
            if isinstance(m, nn.Linear):
                mask = nm_mask(W, N, M, self.dim).to(W.device)

            # Conv2d: weight shape (out_c, in_c, kH, kW)
            elif isinstance(m, nn.Conv2d):
                out_c = W.size(0)
                W_flat = W.view(out_c, -1)  # flatten per-filter
                mask_flat = nm_mask(W_flat, N, M, dim=1).to(W.device)
                mask = mask_flat.view_as(W)

            else:
                # Fallback: group along last dim
                mask = nm_mask(W, N, M, dim=-1).to(W.device)

            # If already pruned, combine new mask with existing one
            if hasattr(m, "weight_mask"):
                # multiply masks (logical AND)
                m.weight_mask.data.mul_(mask)
            else:
                prune.custom_from_mask(m, name="weight", mask=mask)


# =========================
# Utility: finalize pruning (remove re-param)
# =========================
def finalize_pruning(model: nn.Module, module_types=(nn.Linear, nn.Conv2d)):
    """
    Remove pruning re-parametrization and make weights plain tensors with zeros.
    """
    for m in model.modules():
        if isinstance(m, module_types):
            if hasattr(m, "weight_mask"):
                prune.remove(m, "weight")


# =========================
# Example CNN model (Conv + Linear)
# =========================
class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((8, 8))
        self.fc1 = nn.Linear(32 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# =========================
# Example usage
# =========================
if __name__ == "__main__":
    # Create model
    model = SimpleCNN(num_classes=10)

    # Example: compute FLOPs before pruning (if your util supports it)
    # dense_flops_before, sparse_flops_before = compute_model_flops(model)

    # 1) Unstructured magnitude pruning on Conv + Linear
    # prune_method = UnstructuredMagnitudePrune(
    #     amount=0.3,                     # prune 30% each step
    #     module_types=(nn.Linear, nn.Conv2d)
    # )

    # # 2) N:M sparsity pruning (2:4) on Conv + Linear
    # prune_method = NMSparsityPrune(
    #     N=2, M=4,                         # 2:4 pattern
    #     dim=1,                            # for Linear: group along in_features
    #     module_types=(nn.Linear, nn.Conv2d)
    # )
    prune_method = UnstructuredMagnitudePrune(
        module_types=(nn.Linear, nn.Conv2d)
    )

    # Create pruner with L2 regularization on *prunable* weights only
    pruner = Pruner(
        model=model,
        method_str="unstructured_l1_norm",
        reg_lambda=1e-4,
        reg_param_keywords=("weight",),   # only regularize weights of prunable modules
    )

    # Dummy training loop sketch
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
    criterion = nn.CrossEntropyLoss()

    # Fake data loader for demo
    x = torch.randn(1, 3, 32, 32)
    y = torch.randint(0, 10, (32,))
    
    base_ops, base_params = tp.utils.count_ops_and_params(model, example_inputs=(x,))
    print(base_ops, base_params)
    count_model_flops_and_params(model, x, verbose=True)
    # for i in range(200):
    #     pruner.step()
    #     count_model_flops_and_params(model, x, verbose=True)
    pruner.step()
    count_model_flops_and_params(model, x, verbose=True)

    # for epoch in range(3):
    #     optimizer.zero_grad()
    #     out = model(x)
    #     loss = criterion(out, y)
    #     loss.backward()

    #     # Apply L1/L2 gradient regularization ONLY on prunable weights
    #     pruner.regularize()

    #     optimizer.step()

    #     # Optionally prune after each epoch (Conv + Linear)
    #     pruner.step()

    #     conv_sparsity = (model.conv1.weight == 0).float().mean().item()
    #     fc_sparsity = (model.fc1.weight == 0).float().mean().item()
    #     print(
    #         f"Epoch {epoch}, loss={loss.item():.4f}, "
    #         f"conv1 sparsity={conv_sparsity:.3f}, fc1 sparsity={fc_sparsity:.3f}"
    #     )

    # # When done pruning, you can finalize to remove re-param:
    # finalize_pruning(model, module_types=(nn.Linear, nn.Conv2d))

    # # Example: compute FLOPs after pruning (if your util supports it)
    # # dense_flops_after, sparse_flops_after = compute_model_flops(model)
