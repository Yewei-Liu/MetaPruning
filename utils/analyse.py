import math
import os
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt


# =========================================================
# 1. Layer discovery
# =========================================================
def build_layer_infos(model: nn.Module):
    """
    Returns a list of (layer_idx, name, kind) for Conv2d, Linear, BatchNorm2d.
    kind ∈ {"conv", "linear", "bn"}
    """
    layer_infos = []
    idx = 0
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            layer_infos.append((idx, name, "conv"))
            idx += 1
        elif isinstance(m, nn.Linear):
            layer_infos.append((idx, name, "linear"))
            idx += 1
        elif isinstance(m, nn.BatchNorm2d):
            layer_infos.append((idx, name, "bn"))
            idx += 1
    return layer_infos


# =========================================================
# 2. Hooks for activations
# =========================================================
def get_activation_hook(name, storage_dict):
    def hook(module, input, output):
        storage_dict[name] = output.detach()
    return hook


# =========================================================
# 3. Helper: convert weights to 2D matrix for SVD/cond/rank
# =========================================================
def weight_to_matrix(m: nn.Module):
    """
    Convert weight tensor of module m into 2D matrix for SVD/statistics.
    """
    w = m.weight.detach()
    if isinstance(m, nn.Conv2d):
        # (out_channels, in_channels, k, k) -> (out_channels, in_channels * k * k)
        return w.view(w.size(0), -1)
    elif isinstance(m, nn.Linear):
        # (out_features, in_features)
        return w
    elif isinstance(m, nn.BatchNorm2d):
        # Treat gamma as a row vector
        return w.view(1, -1)
    else:
        return w.view(w.size(0), -1)


# =========================================================
# 4. Main analysis function
# =========================================================
def analyze_models(
    models,
    data_loader,
    device,
    num_batches=1,
    bn_gamma_zero_threshold=1e-3,
    entropy_bins=40,
    criterion=None,
):
    """
    Compute per-layer statistics for each model in `models`.

    Args:
        models: list of nn.Module (same architecture), models[0] is reference.
        data_loader: yields (inputs, targets).
        device: torch.device.
        num_batches: how many batches to run for activations/gradients.
        bn_gamma_zero_threshold: threshold for "near-zero" BN gamma.
        entropy_bins: bins for weight entropy histogram.
        criterion: loss function; if None, CrossEntropyLoss is used.

    Metrics (per layer + per model):
      - weight_l1, weight_l2
      - cond_number (condition number from SVD)
      - weight_entropy (histogram-based, deterministic via NumPy)
      - effective_rank (SVD-based)
      - activation_sparsity
      - grad_l2
      - grad_to_weight_ratio
      - taylor_sensitivity (sum |w * grad|)
      - bn_gamma_l1, bn_gamma_l2, bn_beta_l1, bn_beta_l2
      - bn_snr_mean / min / max
      - bn_frac_gamma_near_zero
      - inter_channel_corr_mean / max (Conv weights)

    Returns:
        pandas.DataFrame with one row per (model_idx, layer_idx).
    """
    assert len(models) > 0, "models must be a non-empty list"

    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    ref_model = models[0]
    layer_infos = build_layer_infos(ref_model)

    all_records = []

    for model_idx, model in enumerate(models):
        print(f"Analyzing model {model_idx} ...")

        model = model.to(device)
        model.train()  # want gradients & BN behavior

        # --- activation hooks ---
        activations = {}
        handles = []
        modules = dict(model.named_modules())
        for _, name, kind in layer_infos:
            if name in modules:
                h = modules[name].register_forward_hook(
                    get_activation_hook(name, activations)
                )
                handles.append(h)

        # --- run a few batches to get activations & gradients ---
        model.zero_grad(set_to_none=True)
        batches_used = 0
        for inputs, targets in data_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            model.zero_grad(set_to_none=True)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()

            batches_used += 1
            if batches_used >= num_batches:
                break

        # --- per-layer statistics ---
        for layer_idx, name, kind in layer_infos:
            m = modules[name]
            rec = {
                "model_idx": model_idx,
                "layer_idx": layer_idx,
                "layer_name": name,
                "kind": kind,
            }

            # --------------------------------------------------
            # 4.1. Weight-based metrics
            # --------------------------------------------------
            w = getattr(m, "weight", None)
            if w is not None and w.data is not None:
                w_data = w.detach()
                w_flat = w_data.view(-1)

                # L1 / L2 norms
                l2 = w_flat.norm(2).item()
                l1 = w_flat.norm(1).item()
                rec["weight_l2"] = l2
                rec["weight_l1"] = l1

                # Condition number & effective rank (SVD on CPU for determinism)
                W_mat = weight_to_matrix(m).float().detach().cpu()
                if W_mat.numel() > 0:
                    try:
                        s = torch.linalg.svdvals(W_mat)
                        if s.numel() >= 2 and s[-1] > 1e-8:
                            cond = (s[0] / s[-1]).item()
                        else:
                            cond = float("inf")
                        rec["cond_number"] = cond

                        s_norm = s / (s.sum() + 1e-12)
                        s_pos = s_norm[s_norm > 0]
                        H = -(s_pos * torch.log(s_pos)).sum()
                        eff_rank = torch.exp(H).item()
                        rec["effective_rank"] = eff_rank
                    except RuntimeError:
                        rec["cond_number"] = math.nan
                        rec["effective_rank"] = math.nan
                else:
                    rec["cond_number"] = math.nan
                    rec["effective_rank"] = math.nan

                # Weight entropy (deterministic, CPU + numpy)
                w_np = w_flat.detach().cpu().numpy()
                if w_np.size == 0 or w_np.min() == w_np.max():
                    rec["weight_entropy"] = 0.0
                else:
                    hist, _ = np.histogram(w_np, bins=entropy_bins, density=False)
                    p = hist.astype(np.float64) / (hist.sum() + 1e-12)
                    p = p[p > 0]
                    entropy = float(-(p * np.log(p)).sum())
                    rec["weight_entropy"] = entropy
            else:
                rec["weight_l2"] = math.nan
                rec["weight_l1"] = math.nan
                rec["cond_number"] = math.nan
                rec["effective_rank"] = math.nan
                rec["weight_entropy"] = math.nan

            # --------------------------------------------------
            # 4.2. Activation sparsity
            # --------------------------------------------------
            act = activations.get(name, None)
            if act is not None:
                rec["activation_sparsity"] = (act == 0).float().mean().item()
            else:
                rec["activation_sparsity"] = math.nan

            # --------------------------------------------------
            # 4.3. Gradient-based metrics
            # --------------------------------------------------
            if w is not None and w.grad is not None:
                g_flat = w.grad.detach().view(-1)
                g_l2 = g_flat.norm(2).item()
                rec["grad_l2"] = g_l2
                rec["grad_to_weight_ratio"] = g_l2 / (rec["weight_l2"] + 1e-12)

                # Taylor expansion sensitivity: sum |w * grad|
                if w_flat.numel() == g_flat.numel():
                    taylor = (w_flat * g_flat).abs().sum().item()
                else:
                    taylor = math.nan
                rec["taylor_sensitivity"] = taylor
            else:
                rec["grad_l2"] = math.nan
                rec["grad_to_weight_ratio"] = math.nan
                rec["taylor_sensitivity"] = math.nan

            # --------------------------------------------------
            # 4.4. BatchNorm-specific metrics
            # --------------------------------------------------
            if isinstance(m, nn.BatchNorm2d):
                gamma = m.weight.detach()
                beta = m.bias.detach() if m.bias is not None else None

                rec["bn_gamma_l2"] = gamma.norm(2).item()
                rec["bn_gamma_l1"] = gamma.norm(1).item()
                rec["bn_beta_l2"] = (
                    beta.norm(2).item() if beta is not None else math.nan
                )
                rec["bn_beta_l1"] = (
                    beta.norm(1).item() if beta is not None else math.nan
                )

                # Channel-wise SNR: |gamma| / sqrt(running_var)
                if m.running_var is not None:
                    rv = m.running_var.detach()
                    snr = gamma.abs() / torch.sqrt(rv + 1e-12)
                    rec["bn_snr_mean"] = snr.mean().item()
                    rec["bn_snr_min"] = snr.min().item()
                    rec["bn_snr_max"] = snr.max().item()
                else:
                    rec["bn_snr_mean"] = math.nan
                    rec["bn_snr_min"] = math.nan
                    rec["bn_snr_max"] = math.nan

                frac_zero_gamma = (gamma.abs() < bn_gamma_zero_threshold).float().mean()
                rec["bn_frac_gamma_near_zero"] = frac_zero_gamma.item()
            else:
                rec["bn_gamma_l2"] = math.nan
                rec["bn_gamma_l1"] = math.nan
                rec["bn_beta_l2"] = math.nan
                rec["bn_beta_l1"] = math.nan
                rec["bn_snr_mean"] = math.nan
                rec["bn_snr_min"] = math.nan
                rec["bn_snr_max"] = math.nan
                rec["bn_frac_gamma_near_zero"] = math.nan

            # --------------------------------------------------
            # 4.5. Inter-channel correlations (Conv weights)
            # --------------------------------------------------
            if isinstance(m, nn.Conv2d) and w is not None:
                # shape: (out_channels, in_channels, k, k)
                W_ch = w_data.view(w_data.size(0), -1).detach().cpu()  # (C_out, *)
                if W_ch.size(0) >= 2:
                    W_ch_centered = W_ch - W_ch.mean(dim=1, keepdim=True)
                    try:
                        corr = torch.corrcoef(W_ch_centered)
                        mask = ~torch.eye(
                            corr.size(0), dtype=torch.bool, device=corr.device
                        )
                        off_diag = corr[mask].abs()
                        rec["inter_channel_corr_mean"] = off_diag.mean().item()
                        rec["inter_channel_corr_max"] = off_diag.max().item()
                    except RuntimeError:
                        rec["inter_channel_corr_mean"] = math.nan
                        rec["inter_channel_corr_max"] = math.nan
                else:
                    rec["inter_channel_corr_mean"] = math.nan
                    rec["inter_channel_corr_max"] = math.nan
            else:
                rec["inter_channel_corr_mean"] = math.nan
                rec["inter_channel_corr_max"] = math.nan

            all_records.append(rec)

        # --- remove hooks ---
        for h in handles:
            h.remove()

        # free GPU memory if many models
        torch.cuda.empty_cache()

    df = pd.DataFrame(all_records)
    return df


# =========================================================
# 5. Visualization helpers
# =========================================================

def plot_conv_weight_norms(stats_df, save_dir):
    conv_df = stats_df[stats_df["kind"] == "conv"]

    # L2 norms
    plt.figure(figsize=(10, 5))
    for layer_name in conv_df["layer_name"].unique():
        sub = conv_df[conv_df["layer_name"] == layer_name].sort_values("model_idx")
        plt.plot(sub["model_idx"], sub["weight_l2"], marker="o", label=layer_name)
    plt.xlabel("Model index")
    plt.ylabel("L2 norm (weights)")
    plt.title("Conv layer L2 norms over models")
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc="upper left", fontsize="x-small")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "conv_weight_l2_norms.png"))

    # L1 norms
    plt.figure(figsize=(10, 5))
    for layer_name in conv_df["layer_name"].unique():
        sub = conv_df[conv_df["layer_name"] == layer_name].sort_values("model_idx")
        plt.plot(sub["model_idx"], sub["weight_l1"], marker="o", label=layer_name)
    plt.xlabel("Model index")
    plt.ylabel("L1 norm (weights)")
    plt.title("Conv layer L1 norms over models")
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc="upper left", fontsize="x-small")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "conv_weight_l1_norms.png"))

def plot_conv_norms_by_layer(stats_df, models_to_plot=None, metric="weight_l2", savedir=None):
    """
    Plot L1 or L2 norms for several models on the same axis.
       metric ∈ {"weight_l1", "weight_l2"}
    """
    conv_df = stats_df[stats_df["kind"] == "conv"]

    if models_to_plot is None:
        models_to_plot = sorted(conv_df["model_idx"].unique())

    plt.figure(figsize=(10, 5))
    for mid in models_to_plot:
        sub = conv_df[conv_df["model_idx"] == mid].sort_values("layer_idx")
        plt.plot(sub["layer_idx"], sub[metric], marker="o", label=f"model {mid}")

    plt.xlabel("Layer index")
    plt.ylabel(metric)
    plt.title(f"{metric} vs layer index for multiple models")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    if savedir is not None:
        plt.savefig(os.path.join(savedir, f"conv_{metric}_by_layer.png"))
        
def plot_conv_norm_ratio(stats_df, model_a=1, model_b=0, metric="weight_l2", savedir=None):
    """
    Plot ratio of layer-wise norms:
        ratio = norm(model_a) / norm(model_b)

    Args:
        model_a: numerator model index
        model_b: denominator model index
        metric: "weight_l1" or "weight_l2"
    """
    conv_df = stats_df[stats_df["kind"] == "conv"].copy()

    # Extract model A and B rows
    A = conv_df[conv_df["model_idx"] == model_a].sort_values("layer_idx")
    B = conv_df[conv_df["model_idx"] == model_b].sort_values("layer_idx")

    if len(A) != len(B):
        raise ValueError("Models do not have same conv layer structure.")

    # Compute ratio
    ratio = A[metric].values / (B[metric].values + 1e-12)

    # Plot
    plt.figure(figsize=(10, 4))
    plt.plot(A["layer_idx"], ratio, marker="o")
    plt.xlabel("Layer index")
    plt.ylabel(f"{metric} ratio (model {model_a} / model {model_b})")
    plt.title(f"Layer-wise {metric} ratio: model {model_a} / model {model_b}")
    plt.grid(True)
    plt.tight_layout()
    if savedir is not None:
        plt.savefig(os.path.join(savedir, f"conv_{metric}_ratio_model{model_a}_vs_model{model_b}.png"))


def heatmap_metric(stats_df, kind="conv", metric="weight_l2", log=False):
    sub = stats_df[stats_df["kind"] == kind].copy()
    sub = sub.sort_values(["layer_idx", "model_idx"])
    pivot = sub.pivot(index="layer_name", columns="model_idx", values=metric)

    data = pivot.values
    if log:
        data = np.log10(data + 1e-12)

    plt.figure(figsize=(10, max(4, 0.4 * len(pivot))))
    im = plt.imshow(data, aspect="auto")
    plt.colorbar(im, label=("log10 " if log else "") + metric)
    plt.yticks(np.arange(len(pivot.index)), pivot.index)
    plt.xticks(np.arange(len(pivot.columns)), pivot.columns)
    plt.xlabel("Model index")
    plt.ylabel("Layer")
    plt.title(f"{kind.upper()} {metric} heatmap")
    plt.tight_layout()
    plt.show()


def plot_bn_stats(stats_df):
    bn_df = stats_df[stats_df["kind"] == "bn"].copy()

    # Gamma L2 norm
    plt.figure(figsize=(10, 5))
    for layer_name in bn_df["layer_name"].unique():
        sub = bn_df[bn_df["layer_name"] == layer_name].sort_values("model_idx")
        plt.plot(sub["model_idx"], sub["bn_gamma_l2"], marker="o", label=layer_name)
    plt.xlabel("Model index")
    plt.ylabel("BN gamma L2")
    plt.title("BN gamma L2 norms over models")
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc="upper left", fontsize="x-small")
    plt.tight_layout()
    plt.show()

    # Fraction of near-zero gamma
    plt.figure(figsize=(10, 5))
    for layer_name in bn_df["layer_name"].unique():
        sub = bn_df[bn_df["layer_name"] == layer_name].sort_values("model_idx")
        plt.plot(
            sub["model_idx"],
            sub["bn_frac_gamma_near_zero"],
            marker="o",
            label=layer_name,
        )
    plt.xlabel("Model index")
    plt.ylabel("Fraction |gamma| ≈ 0")
    plt.title("BN sparsity (near-zero gamma) over models")
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc="upper left", fontsize="x-small")
    plt.tight_layout()
    plt.show()

    # SNR mean
    plt.figure(figsize=(10, 5))
    for layer_name in bn_df["layer_name"].unique():
        sub = bn_df[bn_df["layer_name"] == layer_name].sort_values("model_idx")
        plt.plot(sub["model_idx"], sub["bn_snr_mean"], marker="o", label=layer_name)
    plt.xlabel("Model index")
    plt.ylabel("Mean SNR")
    plt.title("BN channel-wise SNR over models")
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc="upper left", fontsize="x-small")
    plt.tight_layout()
    plt.show()


def plot_activation_and_grad(stats_df, kind="conv"):
    sub = stats_df[stats_df["kind"] == kind].copy()

    # Activation sparsity
    plt.figure(figsize=(10, 5))
    for layer_name in sub["layer_name"].unique():
        tmp = sub[sub["layer_name"] == layer_name].sort_values("model_idx")
        plt.plot(tmp["model_idx"], tmp["activation_sparsity"], marker="o", label=layer_name)
    plt.xlabel("Model index")
    plt.ylabel("Activation sparsity")
    plt.title(f"{kind.upper()} activation sparsity over models")
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc="upper left", fontsize="x-small")
    plt.tight_layout()
    plt.show()

    # Gradient L2
    plt.figure(figsize=(10, 5))
    for layer_name in sub["layer_name"].unique():
        tmp = sub[sub["layer_name"] == layer_name].sort_values("model_idx")
        plt.plot(tmp["model_idx"], tmp["grad_l2"], marker="o", label=layer_name)
    plt.xlabel("Model index")
    plt.ylabel("Gradient L2 norm")
    plt.title(f"{kind.upper()} gradient norms over models")
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc="upper left", fontsize="x-small")
    plt.tight_layout()
    plt.show()


def scatter_corr_vs_norm(stats_df):
    conv_df = stats_df[stats_df["kind"] == "conv"].copy()
    if conv_df.empty:
        return
    last_model = conv_df["model_idx"].max()
    sub = conv_df[conv_df["model_idx"] == last_model]

    plt.figure(figsize=(6, 5))
    plt.scatter(sub["weight_l2"], sub["inter_channel_corr_mean"])
    for _, row in sub.iterrows():
        plt.text(row["weight_l2"], row["inter_channel_corr_mean"], row["layer_name"], fontsize=6)
    plt.xlabel("Weight L2 norm")
    plt.ylabel("Mean inter-channel correlation")
    plt.title(f"Conv inter-channel correlation vs L2 (model_idx={last_model})")
    plt.tight_layout()
    plt.show()


def plot_global_taylor_and_ratio(stats_df, kind="conv"):
    sub = stats_df[stats_df["kind"] == kind].copy()
    if sub.empty:
        return
    grouped = sub.groupby("model_idx")

    model_ids = sorted(sub["model_idx"].unique())
    mean_taylor = [grouped.get_group(i)["taylor_sensitivity"].mean() for i in model_ids]
    mean_ratio = [grouped.get_group(i)["grad_to_weight_ratio"].mean() for i in model_ids]

    plt.figure(figsize=(8, 5))
    plt.plot(model_ids, mean_taylor, marker="o", label="mean Taylor sensitivity")
    plt.plot(model_ids, mean_ratio, marker="s", label="mean grad/weight ratio")
    plt.xlabel("Model index")
    plt.title(f"Global {kind.upper()} sensitivity metrics over models")
    plt.legend()
    plt.tight_layout()
    plt.show()

def compare_models_layer_hist(
    models,
    model_idx_a: int,
    model_idx_b: int,
    save_dir: str,
    layer_kinds=("conv", "linear"),
    bins=50,
    log_x=False,
    figsize_mult=1.5,
    xlim=None,
):
    """
    Compare two models by plotting per-filter/per-neuron weight norm histograms layer-by-layer.

    For Conv2d: norm of each filter (out_channel) → sqrt(sum over in × k × k)
    For Linear: norm of each neuron (row) → sqrt(sum over in_features)

    Args:
        models: list of nn.Module
        model_idx_a, model_idx_b: indices of two models to compare
        save_dir: where to save plots
        layer_kinds: tuple of kinds to include, e.g. ("conv", "linear")
        bins: number of histogram bins
        log_x: whether to use log scale on x-axis (helps see small weights)
        figsize_mult: scaling factor for figure size
    """
    os.makedirs(save_dir, exist_ok=True)

    model_a = models[model_idx_a]
    model_b = models[model_idx_b]

    layer_infos_a = build_layer_infos(model_a)
    layer_infos_b = build_layer_infos(model_b)

    # Check layer name alignment (assume same architecture)
    names_a = [(idx, name, kind) for idx, name, kind in layer_infos_a if kind in layer_kinds]
    names_b = [(idx, name, kind) for idx, name, kind in layer_infos_b if kind in layer_kinds]

    if len(names_a) != len(names_b):
        raise ValueError("Model architectures differ in Conv/Linear layer count.")
    for (i, na, ka), (j, nb, kb) in zip(names_a, names_b):
        if na != nb or ka != kb:
            raise ValueError(f"Mismatch at layer {i}: {na}({ka}) vs {nb}({kb})")

    num_layers = len(names_a)
    if num_layers == 0:
        print("No Conv/Linear layers to compare.")
        return

    # Process each layer
    for layer_idx, name, kind in names_a:
        m_a = dict(model_a.named_modules())[name]
        m_b = dict(model_b.named_modules())[name]

        if kind == "conv":
            # Filter-wise L2: (out_c, in_c, k, k) → (out_c,)
            w_a = m_a.weight.detach().cpu()
            w_b = m_b.weight.detach().cpu()
            norms_a = torch.norm(w_a.view(w_a.size(0), -1), dim=1).numpy()
            norms_b = torch.norm(w_b.view(w_b.size(0), -1), dim=1).numpy()
            xlabel = "Filter L2 norm"
        elif kind == "linear":
            # Neuron-wise (row) L2: (out_f, in_f) → (out_f,)
            w_a = m_a.weight.detach().cpu()
            w_b = m_b.weight.detach().cpu()
            norms_a = torch.norm(w_a, dim=1).numpy()
            norms_b = torch.norm(w_b, dim=1).numpy()
            xlabel = "Neuron (row) L2 norm"
        else:
            continue  # skip BN, etc.

        # Skip if empty
        if norms_a.size == 0 or norms_b.size == 0:
            continue

        # Plot
        fig, ax = plt.subplots(1, 1, figsize=(4 * figsize_mult, 3 * figsize_mult))
        
        if xlim is not None:
            ax.set_xlim(xlim)
        
        # Determine shared range
        global_min = min(norms_a.min(), norms_b.min())
        global_max = max(norms_a.max(), norms_b.max())
        if log_x:
            bins_edges = np.logspace(np.log10(max(global_min, 1e-8)), np.log10(global_max + 1e-8), bins + 1)
        else:
            bins_edges = np.linspace(global_min, global_max, bins + 1)

        ax.hist(norms_a, bins=bins_edges, alpha=0.6, label=f"Origin", density=False, edgecolor='k', linewidth=0.5)
        ax.hist(norms_b, bins=bins_edges, alpha=0.6, label=f"Transformed", density=False, edgecolor='k', linewidth=0.5)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Count")
        ax.set_title(f"{name} ({kind.upper()})")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.5)
        if log_x:
            ax.set_xscale("log")

        plt.tight_layout()
        fname = f"hist_{kind}_layer{layer_idx:02d}_{name.replace('.', '_')}.png"
        plt.savefig(os.path.join(save_dir, fname), dpi=150, bbox_inches="tight")
        plt.close(fig)

    print(f"Saved {num_layers} layer-wise histograms to {save_dir}")
    

def plot_inter_channel_corr_by_layer(
    stats_df,
    model_idx=None,
    metric="mean",  # "mean" or "max"
    figsize=(10, 5),
    marker="o",
    linewidth=2,
    title=None,
    xlabel="Layer index",
    ylabel="Inter-channel correlation (absolute)",
    save_path=None,
    ax=None,
):
    """
    Plot raw inter-channel correlation vs. layer index for Conv layers.

    Args:
        stats_df: DataFrame from analyze_models()
        model_idx: int or list of ints; if None, plot all models
        metric: "mean" → use 'inter_channel_corr_mean' (recommended);
                "max"  → use 'inter_channel_corr_max'
        save_path: if provided, saves the figure to this path
    """
    # Filter Conv layers only
    conv_df = stats_df[stats_df["kind"] == "conv"].copy()
    if conv_df.empty:
        print("⚠️ No Conv layers found in stats_df.")
        return None

    # Choose column
    col = f"inter_channel_corr_{metric}"
    if col not in conv_df.columns:
        raise KeyError(f"Column '{col}' not found. Available: {list(conv_df.columns)}")

    # Select models
    if model_idx is None:
        model_indices = sorted(conv_df["model_idx"].unique())
    elif isinstance(model_idx, int):
        model_indices = [model_idx]
    else:
        model_indices = list(model_idx)

    # Prepare plot
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    plotted_any = False
    for mid in model_indices:
        sub = conv_df[conv_df["model_idx"] == mid].sort_values("layer_idx")
        if sub.empty:
            continue

        y = sub[col].values
        x = sub["layer_idx"].values

        # Skip if all NaN
        if np.all(np.isnan(y)):
            print(f"⚠️ All {col} values are NaN for model {mid}. Skipping.")
            continue

        plotted_any = True
        label = f"Model {mid}" if len(model_indices) > 1 else "Inter-channel correlation"
        ax.plot(x, y, marker=marker, linewidth=linewidth, label=label)

        # Optional: annotate points with values (uncomment if desired)
        # for xi, yi in zip(x, y):
        #     if not np.isnan(yi):
        #         ax.text(xi, yi, f"{yi:.2f}", fontsize=7, ha='center', va='bottom')

    if not plotted_any:
        print("❌ No valid data to plot.")
        return None

    # Finalize plot
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title or "Inter-Channel Correlation Across Layers", fontsize=14)
    ax.grid(True, linestyle="--", alpha=0.6)
    if len(model_indices) > 1:
        ax.legend()

    # Set integer ticks for layer indices
    if len(x) > 0:
        ax.set_xticks(x)
        ax.set_xticklabels([str(int(xi)) for xi in x])
        
    ax.tick_params(axis='x', labelsize=4)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"✅ Plot saved to: {save_path}")

    return ax

def plot_effective_rank_by_layer(
    stats_df,
    model_idx=None,
    figsize=(10, 5),
    marker="o",
    linewidth=2,
    title=None,
    xlabel="Layer index",
    ylabel="Effective rank",
    save_path=None,
    ax=None,
    highlight_low_rank=False,
    low_rank_threshold=5.0,
    ylim=None,  # ← added parameter
):
    """
    Plot effective rank vs. layer index for Conv & Linear layers.

    Args:
        stats_df: DataFrame from analyze_models()
        model_idx: int or list of ints; if None, plot all models
        highlight_low_rank: bool, shade regions where eff_rank < threshold
        low_rank_threshold: float, threshold for "low-rank" detection (default: 5.0)
        save_path: if provided, saves figure to this path
        ylim: tuple (ymin, ymax) or None for auto-scaling (default: None)
    """
    # Filter layers where effective_rank is defined (Conv, Linear; BN skipped)
    rank_df = stats_df[
        (stats_df["kind"].isin(["conv", "linear"])) &
        (stats_df["effective_rank"].notna())
    ].copy()

    if rank_df.empty:
        print("⚠️ No layers with valid 'effective_rank' found.")
        return None

    # Select models
    if model_idx is None:
        model_indices = sorted(rank_df["model_idx"].unique())
    elif isinstance(model_idx, int):
        model_indices = [model_idx]
    else:
        model_indices = list(model_idx)

    # Prepare plot
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    plotted_any = False
    all_y_vals = []  # collect all finite y values for auto-scaling
    for mid in model_indices:
        sub = rank_df[rank_df["model_idx"] == mid].sort_values("layer_idx")
        if sub.empty:
            continue

        x = sub["layer_idx"].values
        y = sub["effective_rank"].values

        # Skip if all invalid
        if np.all(np.isnan(y)) or np.all(np.isinf(y)):
            print(f"⚠️ All effective_rank invalid for model {mid}. Skipping.")
            continue

        plotted_any = True
        label = f"Model {mid}" if len(model_indices) > 1 else "Effective rank"
        ax.plot(x, y, marker=marker, linewidth=linewidth, label=label)

        # Collect finite y values for auto-scaling
        finite_y = y[np.isfinite(y)]
        if len(finite_y) > 0:
            all_y_vals.extend(finite_y)

    if not plotted_any:
        print("❌ No valid data to plot.")
        return None

    # Highlight low-rank regions (optional)
    if highlight_low_rank:
        ax.axhspan(0, low_rank_threshold, color='orange', alpha=0.1,
                   label=f'Low-rank zone (<{low_rank_threshold})')
        ax.axhline(low_rank_threshold, color='orange', linestyle='--', linewidth=1)

    # Finalize plot
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title or "Effective Rank Across Layers", fontsize=14)
    ax.grid(True, linestyle="--", alpha=0.6)
    if len(model_indices) > 1:
        ax.legend()

    # Set integer x-ticks
    if len(x) > 0:
        ax.set_xticks(x)
        ax.set_xticklabels([str(int(xi)) for xi in x])

    # Y-axis limits: use user-provided `ylim`, else auto-scale with margin
    if ylim is not None:
        ax.set_ylim(ylim)
    else:
        if all_y_vals:
            ymin = min(all_y_vals)
            ymax = max(all_y_vals)
            y_range = ymax - ymin
            margin = 0.05 * (y_range if y_range > 0 else 1.0)
            ax.set_ylim(
                bottom=max(0, ymin - margin),
                top=ymax + margin
            )
        else:
            ax.set_ylim(bottom=0)

    ax.tick_params(axis='x', labelsize=4)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"✅ Plot saved to: {save_path}")

    return ax



def compare_taylor_sensitivity_hist(
    models,
    model_idx_a: int,
    model_idx_b: int,
    save_dir: str,
    data_loader,
    device,
    criterion=None,
    num_batches=1,
    layer_kinds=("conv", "linear"),
    bins=40,
    figsize=(6, 4),
    density=True,
    show_stats=True,
    xlim=None,
):
    """
    Compare Taylor sensitivity (|w * grad|) distributions between two models, layer-by-layer.

    For each Conv/Linear layer:
      - Compute |w_i * grad_i| for every weight parameter
      - Plot histogram for Model A and Model B side-by-side (same axes)

    Args:
        models: list of nn.Module
        model_idx_a, model_idx_b: indices of models to compare
        save_dir: where to save plots (one PNG per layer)
        data_loader, device: for computing gradients
        num_batches: how many batches to use for gradient estimation
        layer_kinds: which layer types to include
        bins: number of histogram bins
        density: if True, normalize to probability density (area=1)
        show_stats: if True, annotate mean & median
    """
    os.makedirs(save_dir, exist_ok=True)

    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    model_a = models[model_idx_a].to(device)
    model_b = models[model_idx_b].to(device)

    # Get layer info (assume same arch)
    layer_infos = build_layer_infos(model_a)
    layer_infos = [(idx, name, kind) for idx, name, kind in layer_infos if kind in layer_kinds]

    # Compute gradients for both models
    def _compute_grads(model):
        model.train()
        model.zero_grad(set_to_none=True)
        batches_used = 0
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            batches_used += 1
            if batches_used >= num_batches:
                break
        return model

    print(f"Computing gradients for Model {model_idx_a}...")
    model_a = _compute_grads(model_a)
    print(f"Computing gradients for Model {model_idx_b}...")
    model_b = _compute_grads(model_b)

    modules_a = dict(model_a.named_modules())
    modules_b = dict(model_b.named_modules())

    for layer_idx, name, kind in layer_infos:
        m_a = modules_a[name]
        m_b = modules_b[name]

        # Skip if no weight or grad
        if not (hasattr(m_a, 'weight') and m_a.weight is not None and m_a.weight.grad is not None):
            print(f"⚠️ Skipping {name}: missing weight/grad in Model A")
            continue
        if not (hasattr(m_b, 'weight') and m_b.weight is not None and m_b.weight.grad is not None):
            print(f"⚠️ Skipping {name}: missing weight/grad in Model B")
            continue

        # Compute |w * grad| per weight
        w_a = m_a.weight.detach()
        g_a = m_a.weight.grad.detach()
        taylor_a = (w_a * g_a).abs().cpu().numpy().flatten()

        w_b = m_b.weight.detach()
        g_b = m_b.weight.grad.detach()
        taylor_b = (w_b * g_b).abs().cpu().numpy().flatten()

        if taylor_a.size == 0 or taylor_b.size == 0:
            continue

        # Plot
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        if xlim is not None:
            ax.set_xlim(xlim)

        # Shared range for fair comparison
        global_min = min(taylor_a.min(), taylor_b.min())
        global_max = max(taylor_a.max(), taylor_b.max())
        bins_edges = np.linspace(global_min, global_max, bins + 1)

        # Plot histograms
        ax.hist(
            taylor_a,
            bins=bins_edges,
            alpha=0.6,
            label=f"Model {model_idx_a}",
            density=density,
            edgecolor='k',
            linewidth=0.5,
        )
        ax.hist(
            taylor_b,
            bins=bins_edges,
            alpha=0.6,
            label=f"Model {model_idx_b}",
            density=density,
            edgecolor='k',
            linewidth=0.5,
        )

        # Annotate stats (optional)
        if show_stats:
            mean_a, median_a = np.mean(taylor_a), np.median(taylor_a)
            mean_b, median_b = np.mean(taylor_b), np.median(taylor_b)
            ax.axvline(mean_a, color='C0', linestyle='--', linewidth=1, label=f'Mean Origin: {mean_a:.2e}')
            ax.axvline(median_a, color='C0', linestyle=':', linewidth=1, label=f'Median Origin: {median_a:.2e}')
            ax.axvline(mean_b, color='C1', linestyle='--', linewidth=1, label=f'Mean Transformed: {mean_b:.2e}')
            ax.axvline(median_b, color='C1', linestyle=':', linewidth=1, label=f'Median Transformed: {median_b:.2e}')

        # Labels
        ax.set_xlabel("|weight × gradient| (Taylor sensitivity)")
        ax.set_ylabel("Density" if density else "Count")
        ax.set_title(f"{name} ({kind.upper()})\nLayer {layer_idx}")
        ax.legend(fontsize='small')
        ax.grid(True, linestyle="--", alpha=0.5)

        # Save
        safe_name = name.replace(".", "_")
        fname = f"taylor_hist_layer{layer_idx:03d}_{safe_name}_model{model_idx_a}_vs_{model_idx_b}.png"
        plt.savefig(os.path.join(save_dir, fname), dpi=150, bbox_inches="tight")
        plt.close(fig)

    print(f"✅ Saved {len(layer_infos)} Taylor sensitivity histograms to: {save_dir}")
    

def plot_taylor_sensitivity_by_layer(
    stats_df,
    model_idx=None,
    figsize=(10, 5),
    marker="o",
    linewidth=2,
    title=None,
    xlabel="Layer index",
    ylabel="Taylor sensitivity (|w·∇L|)",
    save_path=None,
    ax=None,
    plot_mean=True,
    plot_median=True,
    show_legend=True,
    xlim=None,
):
    """
    Plot mean and/or median Taylor sensitivity vs. layer index.

    Args:
        stats_df: DataFrame from analyze_models()
        model_idx: int or list of ints; if None, plot all models
        plot_mean: bool, whether to plot mean
        plot_median: bool, whether to plot median
        save_path: str, path to save figure (e.g., "taylor_vs_layer.png")
    """
    # Filter layers with Taylor sensitivity (Conv/Linear)
    taylor_df = stats_df[
        (stats_df["kind"].isin(["conv", "linear"])) &
        (stats_df["taylor_sensitivity"].notna())
    ].copy()

    if taylor_df.empty:
        print("⚠️ No layers with valid 'taylor_sensitivity' found.")
        return None

    # Compute per-layer mean & std of |w·g| (if raw stats were stored)
    # But since your current analyze_models only stores SUM (not per-weight),
    # we approximate: 
    #   mean ≈ taylor_sensitivity / num_weights
    #   median ≈ unavailable → we'll skip or estimate if you extend later.

    # However! Your current `taylor_sensitivity` is the *sum* of |w·g| over all weights in the layer.
    # So to get *mean*, we need #weights — reconstruct it:
    def _get_num_weights(row):
        name = row["layer_name"]
        kind = row["kind"]
        model_idx_local = row["model_idx"]
        # Reconstruct layer size from stats (approximate)
        if kind == "conv":
            # weight shape: (out, in, k, k) → stored in weight_l2 context
            # We can't recover exact dim, but if you store it during analysis, better.
            # For now: estimate from weight vector size:
            if not np.isnan(row["weight_l2"]) and row["weight_l2"] > 0:
                w_l2 = row["weight_l2"]
                # rough: mean |w| ≈ w_l2 / sqrt(N) ⇒ N ≈ (w_l2 / mean|w|)^2 — circular.
                # Instead, let's assume you add num_params during analysis (recommended).
                # TEMP: fallback to relative comparison (plot sum, not mean)
                return 1  # so mean = sum
        elif kind == "linear":
            return 1
        return 1

    # ⚠️ Since your current `analyze_models` only stores *sum* (not per-weight),
    # we have two options:
    #   Option 1 (quick): Plot the **sum** (your current `taylor_sensitivity`) — still useful for relative comparison.
    #   Option 2 (better): Modify `analyze_models` to also store `taylor_mean` and `taylor_median` per layer.

    # We'll proceed with **Option 1**: plot the *sum* as a proxy (proportional to mean × N).
    # Add note in title.

    # Select models
    if model_idx is None:
        model_indices = sorted(taylor_df["model_idx"].unique())
    elif isinstance(model_idx, int):
        model_indices = [model_idx]
    else:
        model_indices = list(model_idx)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
        
    if xlim is not None:
        ax.set_xlim(xlim)

    for mid in model_indices:
        sub = taylor_df[taylor_df["model_idx"] == mid].sort_values("layer_idx")
        if sub.empty:
            continue

        x = sub["layer_idx"].values
        y_sum = sub["taylor_sensitivity"].values

        # Skip invalid
        mask = np.isfinite(y_sum)
        x, y_sum = x[mask], y_sum[mask]

        if len(x) == 0:
            continue

        # Plot sum (as proxy for "total layer sensitivity")
        label_base = f"Model {mid}"
        if plot_mean:
            ax.plot(x, y_sum, marker=marker, linewidth=linewidth, 
                    label=f"{label_base} (sum)", linestyle="-")
        # Median not available without per-weight data → skip

    # Finalize
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel + " (sum over layer)", fontsize=12)  # clarify it's sum
    ax.set_title(
        title or "Taylor Sensitivity (|w·∇L|) Across Layers\n(Sum per layer — proportional to mean×size)",
        fontsize=13
    )
    ax.grid(True, linestyle="--", alpha=0.6)
    if show_legend and len(model_indices) > 1:
        ax.legend()

    # Integer ticks
    if len(x) > 0:
        ax.set_xticks(x)
        ax.set_xticklabels([str(int(xi)) for xi in x])
        ax.tick_params(axis='x', labelsize=4)  # smaller x-labels

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"✅ Saved to: {save_path}")

    return ax