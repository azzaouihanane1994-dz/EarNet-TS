"""Resumable controlled training with unseen-identity validation matching."""
from pathlib import Path
import csv
import json
import math
import os
import platform
import random
import shutil
import time
import traceback
from datetime import datetime, timezone
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader, Sampler
from torchvision import transforms as T
from PIL import Image, ImageEnhance, ImageDraw
from .models import EarNetModel, batch_hard_triplet
from .protocols import prepare_disjoint_protocols, audit_protocols, write_json, digest
from .metrics import verification, gallery_matching, centroid_matching, export_plots


BASE = {"architecture": "earnet", "eca": True, "cbam": True,
        "edh": True, "onsa": True, "edh_connected": False,
        "margin_schedule": True, "scale_schedule": True, "margin": .35, "scale": 64.,
        "edh_l1": .2, "edh_eik": .1, "onsa_identity": .05, "onsa_adversary": .05,
        "variance_threshold": 1e-4, "edge_threshold": .25, "eca_kernel": 3,
        "triplet_weight": 0., "triplet_margin": .2, "schedule_epochs": 10}


def corruption(image, kind, severity):
    if severity == 0 or kind == "clean":
        return image
    if kind == "illumination_dark":
        return ImageEnhance.Brightness(image).enhance([1, .75, .5, .25][severity])
    if kind == "illumination_bright":
        return ImageEnhance.Brightness(image).enhance([1, 1.25, 1.5, 1.75][severity])
    if kind == "rotation_2d":
        return image.rotate([0, 10, 20, 30][severity], resample=Image.Resampling.BILINEAR)
    if kind == "occlusion":
        image = image.copy(); w, h = image.size
        fraction = [0, .1, .2, .3][severity]
        width = int(w * fraction)
        ImageDraw.Draw(image).rectangle([w // 2 - width // 2, 0, w // 2 + width // 2, h], fill=(0, 0, 0))
        return image
    raise ValueError(f"Unknown corruption {kind}")


class EarDataset(Dataset):
    def __init__(self, root, items, collection, config, train=False, corruption_kind="clean", severity=0):
        self.root, self.items, self.collection = Path(root), items, collection
        self.label_map = {label: i for i, label in enumerate(sorted({row[1] for row in items}))}
        self.labels = [self.label_map[row[1]] for row in items]
        self.corruption_kind, self.severity = corruption_kind, severity
        size = config.get("input_size", 256)
        operations = []
        if train:
            flip = collection != "ami"
            if config.get("invert_flip", False):
                flip = not flip
            if flip:
                operations.append(T.RandomHorizontalFlip(.5))
            operations.extend([T.RandomRotation(10 if collection == "ami" else 25),
                               T.ColorJitter(brightness=.1 if collection == "ami" else .15,
                                             contrast=.1 if collection == "ami" else .15)])
            if collection != "ami":
                operations.append(T.RandomApply([T.GaussianBlur(3)], p=.15))
        operations.extend([T.Resize((size, size)), T.ToTensor(),
                           T.Normalize([.485, .456, .406], [.229, .224, .225])])
        self.transform = T.Compose(operations)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        with Image.open(self.root / self.items[index][0]) as im:
            image = im.convert("RGB")
            if self.collection == "ami":
                # Avoid the detector's unresolved training lineage in new AMI results.
                w, h = image.size; side = min(w, h)
                image = image.crop(((w-side)//2, (h-side)//2, (w+side)//2, (h+side)//2))
            image = corruption(image, self.corruption_kind, self.severity)
            x = self.transform(image)
        return x, self.labels[index]


class IdentityBatchSampler(Sampler):
    def __init__(self, labels, batch_size, identities_per_batch, seed):
        if batch_size % identities_per_batch or batch_size // identities_per_batch < 2:
            raise ValueError("Identity sampling needs P identities and K>=2 images, batch=P*K.")
        self.groups = {label: np.flatnonzero(np.asarray(labels) == label) for label in set(labels)}
        self.p, self.k = identities_per_batch, batch_size // identities_per_batch
        self.seed, self.steps = seed, math.ceil(len(labels) / batch_size)
        if len(self.groups) < self.p or min(map(len, self.groups.values())) < 2:
            raise ValueError("Too few training identities/images for batch-hard controls.")

    def __len__(self):
        return self.steps

    def __iter__(self):
        rng = np.random.default_rng(self.seed)
        for _ in range(self.steps):
            identities = rng.choice(sorted(self.groups), self.p, replace=False)
            yield [int(index) for label in identities
                   for index in rng.choice(self.groups[label], self.k, replace=len(self.groups[label]) < self.k)]


@torch.inference_mode()
def embeddings(model, dataset, device, workers=0, batch_size=16):
    model.eval(); result, labels = [], []
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers,
                        pin_memory=device.type == "cuda")
    for x, y in loader:
        z, _, _, _ = model(x.to(device, non_blocking=True))
        result.append(z.float().cpu().numpy()); labels.append(y.numpy())
    return np.concatenate(result), np.concatenate(labels)


def calibrated_verification(val_x, val_y, test_x, test_y, config, seed):
    from .metrics import FAR_TARGETS
    val, val_arrays = verification(val_x, val_y, config["max_verification_pairs"], seed)
    test, test_arrays = verification(test_x, test_y, config["max_verification_pairs"], seed)
    calibrated = {}
    for target in FAR_TARGETS:
        threshold = val["test_roc_operating_points"][str(target)]["threshold"]
        if threshold is None:
            calibrated[str(target)] = {"threshold": None, "tar_percent": 0., "achieved_far_fraction": 0.}
        else:
            calibrated[str(target)] = {"threshold": threshold,
                                       "tar_percent": float(100 * np.mean(test_arrays["genuine_scores"] >= threshold)),
                                       "achieved_far_fraction": float(np.mean(test_arrays["impostor_scores"] >= threshold))}
    test["validation_calibrated_operating_points"] = calibrated
    return test, test_arrays, val_arrays


def save_checkpoint(path, value):
    temp = Path(path).with_suffix(".tmp")
    torch.save(value, temp); temp.replace(path)


def train_job(root, output, job):
    config, collection, seed = job["config"], job["collection"], job["seed"]
    device = torch.device(config.get("device", "cuda"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA build/device unavailable; full training will not silently fall back to CPU.")
    destination = Path(output) / "runs" / job["id"]
    destination.mkdir(parents=True, exist_ok=True)
    protocol_path = Path(output) / "protocols" / f"{collection}_seed{seed}.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if not protocol["subject_disjoint"] or not protocol["independent_test"]:
        raise ValueError("Primary training requires the separate disjoint independent protocol.")
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    train = EarDataset(root, protocol["items"]["train"], collection, config, train=True)
    validation = EarDataset(root, protocol["items"]["val"], collection, config)
    test = EarDataset(root, protocol["items"]["test"], collection, config)
    model = EarNetModel(root, len(train.label_map), config, pretrained=True).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    scaler = torch.amp.GradScaler("cuda", init_scale=1024,
                                 enabled=device.type == "cuda" and config["mixed_precision"])
    best, best_epoch, start, elapsed_training = -1., 0, 1, 0.
    last = destination / "last.pt"
    if last.exists():
        checkpoint = torch.load(last, map_location=device, weights_only=True)
        if checkpoint["config"] != config or checkpoint["protocol_sha256"] != digest(protocol_path):
            raise ValueError("Resume configuration/protocol differs from its checkpoint.")
        model.load_state_dict(checkpoint["model"]); optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        best, best_epoch = checkpoint["best"], checkpoint["best_epoch"]
        start, elapsed_training = checkpoint["epoch"] + 1, checkpoint["elapsed_training_seconds"]
        torch.set_rng_state(checkpoint["torch_rng"].cpu())
        if device.type == "cuda":
            torch.cuda.set_rng_state_all([state.cpu() for state in checkpoint["cuda_rng"]])
    write_json(destination / "configuration.json", job)
    curves = []
    curve_path = destination / "curves.json"
    if curve_path.exists():
        curves = json.loads(curve_path.read_text(encoding="utf-8"))[:start-1]
    if curves and start - 1 >= config["minimum_epochs"] and start - 1 - best_epoch >= config["early_stopping_patience"]:
        start = config["epochs"] + 1
    for epoch in range(start, config["epochs"] + 1):
        began = time.perf_counter()
        model.train(); model.set_epoch(epoch)
        sampler = IdentityBatchSampler(train.labels, config["batch_size"], config["identities_per_batch"], seed + 10000 * epoch)
        generator = torch.Generator().manual_seed(seed + epoch)
        loader = DataLoader(train, batch_sampler=sampler, num_workers=config["num_workers"],
                            pin_memory=device.type == "cuda", generator=generator)
        optimizer.zero_grad(set_to_none=True)
        loss_sum, correct, seen, skipped_updates = 0., 0, 0, 0
        accumulation = config["gradient_accumulation"]
        for step, (x, y) in enumerate(loader):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device.type, enabled=scaler.is_enabled()):
                z, logits, _, terms = model(x, y, losses=True)
                loss = F.cross_entropy(logits.float(), y) + sum(terms.values())
                if config.get("triplet_weight", 0):
                    loss += config["triplet_weight"] * batch_hard_triplet(z.float(), y, config["triplet_margin"])
                group_size = min(accumulation, len(loader) - (step // accumulation) * accumulation)
                scaled_loss = loss / group_size
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite loss in {job['id']} epoch {epoch}, step {step}.")
            scaler.scale(scaled_loss).backward()
            if (step + 1) % accumulation == 0 or step + 1 == len(loader):
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 5.)
                previous_scale = scaler.get_scale()
                scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
                skipped_updates += int(scaler.get_scale() < previous_scale)
            with torch.no_grad():
                prediction = (z @ F.normalize(model.head.weight, dim=1).T).argmax(1)
                correct += int((prediction == y).sum()); seen += len(y)
            loss_sum += float(loss.detach()) * len(y)
            if step == 0 or (step + 1) % 100 == 0:
                print(f"{job['id']} epoch={epoch} batch={step+1}/{len(loader)} loss={float(loss.detach()):.4f}", flush=True)
        val_x, val_y = embeddings(model, validation, device, config["num_workers"])
        val_gallery, _ = gallery_matching(val_x, val_y, bootstrap=100, seed=seed)
        val_rank1 = val_gallery["ranks_percent"]["1"]
        seconds = time.perf_counter() - began; elapsed_training += seconds
        curves.append({"epoch": epoch, "loss": loss_sum / seen, "train_classifier_accuracy_percent": 100 * correct / seen,
                       "unseen_validation_gallery_rank1_percent": val_rank1,
                       "margin": model.head.m, "scale": model.head.s, "epoch_seconds": seconds,
                       "amp_skipped_updates": skipped_updates})
        if val_rank1 > best:
            best, best_epoch = val_rank1, epoch
            save_checkpoint(destination / "best.pt", {"model": model.state_dict(), "epoch": epoch,
                                                       "validation_rank1_percent": best, "config": config,
                                                       "protocol_sha256": digest(protocol_path)})
        save_checkpoint(last, {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
                               "epoch": epoch, "best": best, "best_epoch": best_epoch, "config": config,
                               "protocol_sha256": digest(protocol_path), "elapsed_training_seconds": elapsed_training,
                               "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else []})
        write_json(curve_path, curves)
        print(f"{job['id']} epoch={epoch} loss={loss_sum/seen:.4f} unseen-val-R1={val_rank1:.2f}% seconds={seconds:.1f}", flush=True)
        write_json(Path(output) / "heartbeat.json", {"job": job["id"], "epoch": epoch, "best_epoch": best_epoch,
                                                    "best_validation_rank1_percent": best,
                                                    "utc": datetime.now(timezone.utc).isoformat()})
        if epoch >= config["minimum_epochs"] and epoch - best_epoch >= config["early_stopping_patience"]:
            break
    best_ckpt = torch.load(destination / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(best_ckpt["model"])
    val_x, val_y = embeddings(model, validation, device, config["num_workers"])
    if job["group"] == "sensitivity":
        result = {"status": "completed", "job": job["id"], "collection": collection,
                  "variant": job["variant"], "group": "sensitivity", "seed": seed,
                  "protocol_sha256": digest(protocol_path),
                  "checkpoint_sha256": digest(destination / "best.pt"),
                  "selected_epoch": best_epoch, "epochs_executed": curves[-1]["epoch"],
                  "training_seconds": elapsed_training,
                  "scope": "validation-only parameter selection; test not evaluated",
                  "validation_gallery": gallery_matching(val_x, val_y, seed=seed)[0]}
        np.savez_compressed(destination / "embeddings.npz", validation_embeddings=val_x, validation_labels=val_y)
        write_json(destination / "metrics.json", result)
        return result
    test_x, test_y = embeddings(model, test, device, config["num_workers"])
    gallery, gallery_arrays = gallery_matching(test_x, test_y, seed=seed)
    verification_result, scores, validation_scores = calibrated_verification(val_x, val_y, test_x, test_y, config, seed)
    np.savez_compressed(destination / "embeddings.npz", validation_embeddings=val_x, validation_labels=val_y,
                        test_embeddings=test_x, test_labels=test_y)
    np.savez_compressed(destination / "scores.npz", **scores, **gallery_arrays)
    np.savez_compressed(destination / "validation_scores.npz", **validation_scores)
    result = {"status": "completed", "job": job["id"], "collection": collection, "variant": job["variant"],
              "group": job["group"], "seed": seed, "protocol_sha256": digest(protocol_path),
              "checkpoint_sha256": digest(destination / "best.pt"), "selected_epoch": best_epoch,
              "epochs_executed": curves[-1]["epoch"], "training_seconds": elapsed_training,
              "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated() if device.type == "cuda" else None,
              "gallery": gallery, "verification": verification_result,
              "training_parameters": sum(p.numel() for p in model.parameters()),
              "inference_parameters": sum(p.numel() for p in model.backbone.parameters()),
              "protocol": protocol["protocol"], "dataset_label": protocol["dataset_label"],
              "preprocessing": protocol["preprocessing"],
              "detector_identity_provenance": protocol["detector_identity_provenance"],
              "initialization": "scratch transformer" if model.backbone.vit else "ImageNet ResNet-50 only",
              "environment": {"torch": torch.__version__, "python": platform.python_version(),
                              "device": str(device), "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None}}
    export_plots(destination, scores)
    from .analysis import robustness_and_attention
    if job["variant"] == "earnet" and seed == 42 and config.get("run_analysis_after_training", True):
        robustness_and_attention(model, root, protocol, config, device, destination)
    # A completion record is written only after every required output succeeds.
    write_json(destination / "metrics.json", result)
    del model, optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


