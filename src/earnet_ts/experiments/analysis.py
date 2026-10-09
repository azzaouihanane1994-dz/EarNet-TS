"""Attention and synthetic-corruption evaluation."""
from pathlib import Path
import numpy as np
import torch
from .protocols import write_json, gallery_indices
from .metrics import gallery_matching, verification

def attention_export(model, root, items, collection, config, device, destination):
    from torch.utils.data import DataLoader
    from .training import EarDataset
    from .protocols import gallery_indices
    g, _ = gallery_indices([row[1] for row in items])
    rng = np.random.default_rng(42)
    chosen = np.sort(rng.choice(g, min(64, len(g)), replace=False))
    selected = [items[int(index)] for index in chosen]
    dataset = EarDataset(root, selected, collection, config)
    data, labels = [], []
    for x, y in DataLoader(dataset, batch_size=8, shuffle=False):
        _, _, aux, _ = model(x.to(device))
        weights = aux.get("eca_w")
        if weights is None:
            return None
        data.append(weights.flatten(1).cpu().numpy()); labels.extend(y.tolist())
    weights = np.concatenate(data)
    destination = Path(destination); destination.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination / "eca_weights.npz", weights=weights, indices=chosen)
    write_json(destination / "eca_samples.json", selected)
    stream_means = np.column_stack([weights[:, :1024].mean(1), weights[:, 1024:].mean(1)])
    draws = rng.integers(len(weights), size=(1000, len(weights)))
    ci = np.percentile(stream_means[draws].mean(1), [2.5, 97.5], axis=0)
    summary = {"samples": len(weights), "rgb_channels": [1, 1024], "contour_channels": [1025, 2048],
               "mean_weight": stream_means.mean(0).tolist(), "bootstrap_95": ci.tolist(),
               "interpretation": "Channel-weight magnitudes are descriptive, not causal modality contributions."}
    write_json(destination / "eca_summary.json", summary)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.3))
    axes[0].plot(np.arange(1, 2049), weights.mean(0), lw=.8)
    axes[0].axvline(1024.5, color="black", ls="--")
    axes[0].set(xlabel="Concatenated channel (RGB 1–1024; contour 1025–2048)", ylabel="Mean ECA weight")
    axes[1].boxplot([stream_means[:, 0], stream_means[:, 1]], tick_labels=["RGB", "Contour"])
    axes[1].set(ylabel="Per-image mean weight")
    fig.tight_layout(); fig.savefig(destination / "eca_streams.png", dpi=220); plt.close(fig)
    return summary

def robustness_and_attention(model, root, protocol, config, device, destination):
    from .training import EarDataset, embeddings, calibrated_verification
    model.eval()
    clean_dataset = EarDataset(root, protocol["items"]["test"], protocol["collection"], config)
    clean_x, clean_y = embeddings(model, clean_dataset, device, config["num_workers"])
    validation = EarDataset(root, protocol["items"]["val"], protocol["collection"], config)
    val_x, val_y = embeddings(model, validation, device, config["num_workers"])
    g, p = gallery_indices(clean_y)
    results = []
    for kind in ["illumination_dark", "illumination_bright", "occlusion", "rotation_2d"]:
        for severity in config["robustness_severities"]:
            if severity == 0:
                x = clean_x.copy()
            else:
                dataset = EarDataset(root, protocol["items"]["test"], protocol["collection"], config,
                                        corruption_kind=kind, severity=severity)
                x, _ = embeddings(model, dataset, device, config["num_workers"])
                x[g] = clean_x[g]
            gallery, _ = gallery_matching(x, clean_y, bootstrap=1000)
            verif, _, _ = calibrated_verification(val_x, val_y, x, clean_y, config, protocol["seed"])
            results.append({"corruption": kind, "severity": severity, "gallery": gallery, "verification": verif})
    write_json(Path(destination) / "robustness.json", {"records": results,
               "scope": "synthetic image corruptions; clean enrollment, corrupted probes",
               "pose_status": "2D rotation only; real annotated 3D pose not available",
               "side_status": "no authenticated anatomical left/right annotations supplied"})
    attention_export(model, root, protocol["items"]["test"], protocol["collection"], config, device, destination)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 3.6))
    for kind in ["illumination_dark", "illumination_bright", "occlusion", "rotation_2d"]:
        rows = [r for r in results if r["corruption"] == kind]
        ax.plot([r["severity"] for r in rows], [r["gallery"]["ranks_percent"]["1"] for r in rows], marker="o", label=kind)
    ax.set(xlabel="Predeclared synthetic severity", ylabel="Gallery Rank-1 (%)", xticks=[0, 1, 2, 3])
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(Path(destination) / "robustness.png", dpi=220); plt.close(fig)
