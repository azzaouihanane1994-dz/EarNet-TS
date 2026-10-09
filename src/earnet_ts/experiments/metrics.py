"""One score-to-metric pipeline, with distinct pairs and explicit units."""
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score
from .protocols import gallery_indices, digest, write_json

FAR_TARGETS = (0.1, 0.05, 0.01, 0.001)


def normalized(embeddings):
    x = np.asarray(embeddings, dtype=np.float64)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError("Embeddings must be a finite matrix.")
    lengths = np.linalg.norm(x, axis=1, keepdims=True)
    if np.any(lengths < 1e-12):
        raise ValueError("Zero-length embedding.")
    return x / lengths


def unrank_pairs(index, n):
    q = np.asarray(index, dtype=np.int64)
    first = np.floor(((2 * n - 1) - np.sqrt((2 * n - 1) ** 2 - 8 * q)) / 2).astype(np.int64)
    start = first * (2 * n - first - 1) // 2
    return np.column_stack([first, q - start + first + 1])


def distinct_pairs(labels, maximum=100000, seed=42):
    labels = np.asarray(labels)
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(labels == label) for label in np.unique(labels)]
    counts = np.array([len(g) * (len(g) - 1) // 2 for g in groups], dtype=np.int64)
    cumulative = np.cumsum(counts)
    total_positive = int(cumulative[-1])
    total = len(labels) * (len(labels) - 1) // 2
    total_negative = total - total_positive
    if not total_positive or not total_negative:
        raise ValueError("Verification requires genuine and impostor pairs.")
    q = np.sort(rng.choice(total_positive, min(maximum, total_positive), replace=False))
    positive = []
    previous = 0
    for g, end in zip(groups, cumulative):
        selected = q[(q >= previous) & (q < end)] - previous
        if len(selected):
            positive.append(g[unrank_pairs(selected, len(g))])
        previous = end
    positive = np.concatenate(positive)
    if total <= 250000:
        all_pairs = unrank_pairs(np.arange(total), len(labels))
        negative = all_pairs[labels[all_pairs[:, 0]] != labels[all_pairs[:, 1]]]
        if len(negative) > maximum:
            negative = negative[rng.choice(len(negative), maximum, replace=False)]
    else:
        target, chosen = min(maximum, total_negative), set()
        while len(chosen) < target:
            candidates = rng.integers(total, size=min(total, 2 * (target - len(chosen)) + 100))
            pairs = unrank_pairs(candidates, len(labels))
            chosen.update(int(i) for i, pair in zip(candidates, pairs)
                          if labels[pair[0]] != labels[pair[1]])
        q = np.array(sorted(chosen), dtype=np.int64)
        if len(q) > target:
            q = q[rng.choice(len(q), target, replace=False)]
        negative = unrank_pairs(q, len(labels))
    return positive, negative


def roc_metrics(genuine, impostor):
    genuine, impostor = np.asarray(genuine), np.asarray(impostor)
    truth = np.r_[np.ones(len(genuine)), np.zeros(len(impostor))]
    scores = np.r_[genuine, impostor]
    far, tar, thresholds = roc_curve(truth, scores, drop_intermediate=False)
    difference = far - (1 - tar)
    hi = int(np.flatnonzero(difference >= 0)[0])
    lo = max(0, hi - 1)
    ratio = 0.0 if hi == lo or difference[hi] == difference[lo] else -difference[lo] / (difference[hi] - difference[lo])
    eer = far[lo] + ratio * (far[hi] - far[lo])
    operating = {}
    for target in FAR_TARGETS:
        eligible = np.flatnonzero(far <= target)
        index = int(eligible[np.argmax(tar[eligible])])
        operating[str(target)] = {
            "far_fraction": float(far[index]), "tar_percent": float(100 * tar[index]),
            "threshold": float(thresholds[index]) if np.isfinite(thresholds[index]) else None,
        }
    return {"auc": float(roc_auc_score(truth, scores)), "eer_percent": float(100 * eer),
            "eer_convention": "linear interpolation of empirical FAR/FRR crossing",
            "genuine_pairs": len(genuine), "impostor_pairs": len(impostor),
            "test_roc_operating_points": operating}, (far, tar, thresholds)


def cluster_bootstrap_intervals(labels, positive, negative, genuine, impostor, repetitions=200, seed=42):
    labels = np.asarray(labels)
    identities, indices = np.unique(labels, return_inverse=True)
    truth = np.r_[np.ones(len(genuine), dtype=bool), np.zeros(len(impostor), dtype=bool)]
    scores = np.r_[genuine, impostor]
    order = np.argsort(-scores, kind="stable")
    sorted_scores, sorted_truth = scores[order], truth[order]
    ends = np.r_[np.flatnonzero(np.diff(sorted_scores)), len(scores)-1]
    rng = np.random.default_rng(seed)
    collected = {"auc": [], "eer_percent": [], **{f"tar_at_{far}": [] for far in FAR_TARGETS}}
    for _ in range(repetitions):
        counts = np.bincount(rng.integers(len(identities), size=len(identities)), minlength=len(identities))
        pos_w = counts[indices[positive[:, 0]]].astype(float)
        neg_w = (counts[indices[negative[:, 0]]] * counts[indices[negative[:, 1]]]).astype(float)
        weights = np.r_[pos_w, neg_w][order]
        if pos_w.sum() == 0 or neg_w.sum() == 0:
            continue
        tar = np.r_[0., np.cumsum(weights * sorted_truth)[ends] / pos_w.sum()]
        far = np.r_[0., np.cumsum(weights * ~sorted_truth)[ends] / neg_w.sum()]
        difference = far - (1-tar)
        hi = int(np.flatnonzero(difference >= 0)[0]); lo = max(0, hi-1)
        ratio = 0. if hi == lo or difference[hi] == difference[lo] else -difference[lo]/(difference[hi]-difference[lo])
        collected["auc"].append(float((np.trapezoid(tar, far) if hasattr(np, "trapezoid") else np.trapz(tar, far))))
        collected["eer_percent"].append(float(100*(far[lo]+ratio*(far[hi]-far[lo]))))
        for target in FAR_TARGETS:
            collected[f"tar_at_{target}"].append(float(100*tar[far<=target].max()))
    return {"method": "identity cluster resampling; within-identity pair weight c_i, cross-identity pair weight c_i*c_j",
            "repetitions": repetitions, "valid_draws": len(collected["auc"]),
            "interval95": {name: np.percentile(values, [2.5, 97.5]).tolist() for name, values in collected.items() if values},
            "scope": "conditional on fixed trained embeddings; not training/split uncertainty"}


def verification(embeddings, labels, maximum=100000, seed=42, bootstrap=200):
    x = normalized(embeddings)
    positive, negative = distinct_pairs(labels, maximum, seed)
    scores = [np.einsum("ij,ij->i", x[p[:, 0]], x[p[:, 1]]) for p in [positive, negative]]
    result, curve = roc_metrics(*scores)
    if bootstrap:
        result["identity_bootstrap"] = cluster_bootstrap_intervals(labels, positive, negative, *scores,
                                                                  repetitions=bootstrap, seed=seed)
    return result, {"genuine_pairs": positive, "impostor_pairs": negative,
                    "genuine_scores": scores[0], "impostor_scores": scores[1],
                    "far": curve[0], "tar": curve[1], "thresholds": curve[2]}


def gallery_matching(embeddings, labels, bootstrap=1000, seed=42):
    x, labels = normalized(embeddings), np.asarray(labels)
    g, p = gallery_indices(labels)
    similarities = x[p] @ x[g].T
    order = np.argsort(-similarities, axis=1, kind="stable")
    correct = labels[g][order] == labels[p, None]
    ranks = np.argmax(correct, axis=1) + 1
    result = {"gallery_images": len(g), "probe_images": len(p),
              "enrolled_identities": len(np.unique(labels[g])),
              "ranks_percent": {str(k): float(100 * np.mean(ranks <= k)) for k in [1, 5, 10]}}
    unique = np.unique(labels[p])
    success = np.array([np.sum(ranks[labels[p] == i] == 1) for i in unique])
    size = np.array([np.sum(labels[p] == i) for i in unique])
    rng = np.random.default_rng(seed)
    samples = rng.integers(len(unique), size=(bootstrap, len(unique)))
    distribution = 100 * success[samples].sum(1) / size[samples].sum(1)
    result["rank1_identity_bootstrap_95_percent"] = np.percentile(distribution, [2.5, 97.5]).tolist()
    return result, {"gallery_indices": np.asarray(g), "probe_indices": np.asarray(p),
                    "probe_ranks": ranks, "similarities": similarities}


def centroid_matching(train, train_labels, test, test_labels):
    train, test = normalized(train), normalized(test)
    train_labels, test_labels = np.asarray(train_labels), np.asarray(test_labels)
    identities = np.unique(train_labels)
    centroids = normalized(np.array([train[train_labels == i].mean(0) for i in identities]))
    known = np.isin(test_labels, identities)
    order = np.argsort(-(test[known] @ centroids.T), axis=1, kind="stable")
    ranks = identities[order] == test_labels[known, None]
    return {"known_queries": int(known.sum()), "unknown_queries": int((~known).sum()),
            "centroids": len(identities), "topk_percent": {
                str(k): float(100 * np.mean(ranks[:, :min(k, len(identities))].any(1)))
                for k in [1, 5, 10]}}


def export_plots(output, verification_arrays, gallery_arrays=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    output = Path(output)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    far, tar = verification_arrays["far"], verification_arrays["tar"]
    axes[0].semilogx(np.maximum(far, 1e-6), 100 * tar)
    axes[0].set(xlabel="FAR (fraction)", ylabel="TAR (%)", xlim=(1e-4, 1), ylim=(0, 100))
    for name in ["genuine", "impostor"]:
        axes[1].hist(verification_arrays[name + "_scores"], bins=40, density=True, alpha=.5, label=name)
    axes[1].set(xlabel="Cosine similarity", ylabel="Density")
    axes[1].legend()
    fig.tight_layout(); fig.savefig(output / "verification.png", dpi=220); plt.close(fig)


def reconcile_saved(root, output):
    root, output = Path(root), Path(output)
    all_results = {}
    for collection in ["ami", "awe", "earvn10", "uerc2019"]:
        source = root / "results/main" / collection / "embeddings"
        arrays = {name: np.load(source / (name + ".npy"), allow_pickle=False)
                  for name in ["train_embeddings", "train_labels", "test_embeddings", "test_labels"]}
        x, y = arrays["test_embeddings"], arrays["test_labels"]
        metrics, scores = verification(x, y)
        gallery, gallery_arrays = gallery_matching(x, y)
        centroids = centroid_matching(arrays["train_embeddings"], arrays["train_labels"], x, y)
        destination = output / "reconciled" / collection
        destination.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(destination / "scores.npz", **scores, **gallery_arrays)
        result = {"collection": collection, "status": "recomputed_from_saved_embeddings",
                  "scope": "historical known-identity protocol; not a new independent training/test run",
                  "source_hashes": {name: digest(source / (name + ".npy")) for name in arrays},
                  "verification": metrics, "identification": gallery, "centroid_matching": centroids}
        if collection == "ami":
            result["warning"] = "Historical validation/test overlap remains; test/train labels differ in coverage."
        write_json(destination / "metrics.json", result)
        export_plots(destination, scores, gallery_arrays)
        all_results[collection] = result
    write_json(output / "reconciled_metrics.json", all_results)
    return all_results
