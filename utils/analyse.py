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
