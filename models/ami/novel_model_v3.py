# novel_model_v3.py
import os, json, random, math, io, sys
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from glob import glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import cv2
from torchvision import transforms as TVT
from torchvision.models import resnet50, ResNet50_Weights
from torchvision import utils as vutils

from PIL import Image as PILImage, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import logging
logging.getLogger("albumentations").setLevel(logging.ERROR)
logging.getLogger("albumentations.check_version").setLevel(logging.ERROR)
class _SilenceAlbVer(logging.Filter):
    def filter(self, record):
        return not record.name.startswith("albumentations.check_version")
logging.getLogger().addFilter(_SilenceAlbVer())
import re
# Optional: Ultralytics YOLO
try:
    from ultralytics import YOLO
    _HAS_YOLO = True
except Exception:
    _HAS_YOLO = False

# Albumentations preferred; fallback to torchvision if missing
try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    _HAS_ALB = True
except Exception:
    _HAS_ALB = False
    A = None
    ToTensorV2 = None

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score
import pandas as pd

import json, itertools, math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# ---------------------
# Configuration
# ---------------------
CONFIG = {
    "pretrained_ckpt": "./AWE_model/best.ckpt",
    "dataset_dir": "../datasets/AMI/AMI",
    "work_dir": "./AMI_model/results/100epochs_novel_model_v3",
    "seed": 42,

    # Image / model
    "input_size": 256,
    "embedding_dim": 512,
    "arcface_s": 64.0,
    "arcface_m": 0.35,

    # Backbone choice
    # "resnet" -> single-stream ResNet50 (legacy)
    # "two_stream" -> RGB vs. Contour with ECA fusion @ stage3
    "backbone_type": "two_stream",

    # Attention
    "use_attention": True,            # CBAM after layer4 (optional)
    "attention_type": "cbam",
    "eca_k": 3,                       # ECA 1D conv kernel for fusion

    # EDH (Eikonal Distance Head) auxiliary
    "edh": {
        "enabled": True,
        "l1_weight": 0.1,             # L1 to pseudo SDT
        "eik_weight": 0.05,            # eikonal |∇d|-1
        "downsample": 4               # predict at 1/downsample spatial res
    },

    # ONSA (Occluder Null-Space Adversary)
    "onsa": {
        "enabled": True,
        "weight": 0.02,               # weight for (z·u)^2 penalty (backbone)
        "adv_weight": 0.02,           # weight for adversary to maximize (z·u)^2
        "grl": 1.0                    # not used (explicit detach-based min-max)
    },

    # Training
    "epochs": 100,
    "batch_size": 8,
    "lr": 3e-4,
    "weight_decay": 1e-4,
    "num_workers": 0,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "fp16": True,  # mixed precision on CUDA

    # Splits & caps
    "train_frac": 0.7,
    "val_frac": 0.1,         # test = 1 - train - val
    "max_images_per_person": 100,  # per split
    "min_images_person": 5,

    # Cropping / detection
    "use_yolo_crop": True,
    "yolo_conf": 0.25,
    "yolo_iou": 0.50,
    "yolo_pad": 0.10,
    "yolo_square": False,
    "force_recache": True,
    "ear_detector_weights": "./AMI_model/yolov8s_ear_detection.pt",
    "crop_cache_dir": "./AMI_model/cache/",
    "crop_square_pad": 0.15,         # fraction of max side

    # Augmentations (train only)
    "aug": {
        "hflip": False,
        "vflip": False,
        "rotate_deg": 10,      # was 25
        "brightness": 0.10,
        "contrast": 0.10,
        "blur_prob": 0.0,      # off for AMI
        "jpeg_prob": 0.05,
        "perspective_prob": 0.05,
    },

    "debug_samples": {
        "enabled": True,
        "max_save": 48,
        "out_dir": "samples",
        "save_every_n_batches": 1
    },

    # Eval / export
    "save_every": 1,
    "topk": [1, 5, 10],

    # Verification toggle + params
    "do_verification": True,
    "verification": {
        "max_genuine_pairs": 100000,
        "max_impostor_pairs": 100000,
        "far_targets": [1e-1, 5e-2, 1e-2, 1e-3],
        "save_test_test_similarity": True,
        "save_test_centroid_similarity": True
    },

    # Gallery/Probe protocol on TEST split
    "gallery_probe": {
        "enabled": True,
        "gallery_per_class": 1,
        "selection": "first",     # or "random"
        "seed": 42,
        "use_train_as_gallery": True,   # <— enables fallback
    },

    # Open-set & calibration (ported from v2)
    "open_set": {
        "enabled": True,
        "temperature_calibration": True,
        # percentile of CORRECTLY classified val MSP for unknown rejection
        "val_threshold_percentile": 10,
        # percentile on correct-val centroid similarity
        "cosine_threshold_percentile": 5
    },

    # Visualization exports (tight ablations)
    "viz": {
        "enabled": True,
        "max_batches": 2,     # how many batches to export from test_dl
        "out_dir": "viz"
    }
}

# ---------------------
# Utils
# ---------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def list_images(person_dir: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    return [p for p in sorted(person_dir.glob("*")) if p.suffix.lower() in exts]

def safe_load_image(path: Path) -> PILImage.Image:
    with PILImage.open(path) as im:
        im = im.convert("RGB")
        return im.copy()

def ensure_dir(d: Path | str):
    Path(d).mkdir(parents=True, exist_ok=True)

def crop_with_box(img: PILImage.Image, box, pad_ratio=0.15) -> PILImage.Image:
    w, h = img.size
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    pad = int(pad_ratio * max(bw, bh))
    x1p = max(0, x1 - pad)
    y1p = max(0, y1 - pad)
    x2p = min(w, x2 + pad)
    y2p = min(h, y2 + pad)
    return img.crop((x1p, y1p, x2p, y2p))

# ---------------------
# Optional YOLO ear detector
# ---------------------
class EarDetector:
    def __init__(self, weights_path: str,
                 conf: float = 0.25,
                 iou: float = 0.50,
                 pad: float = 0.02,
                 square: bool = False,
                 device: Optional[str] = None):
        self.enabled = False
        self.model = None
        self.conf = float(conf)
        self.iou = float(iou)
        self.pad = float(pad)
        self.square = bool(square)
        self.device = device
        if weights_path and _HAS_YOLO and Path(weights_path).exists():
            try:
                self.model = YOLO(weights_path)
                if self.device:
                    try:
                        self.model.to(self.device)
                    except Exception:
                        pass
                self.enabled = True
                print("[INFO] YOLO ear detector ENABLED.")
            except Exception as e:
                print(f"[WARN] YOLO init failed: {e}. Falling back to center-crop.")
        else:
            if weights_path and not _HAS_YOLO:
                print("[WARN] ultralytics not installed. Install to enable ear detection.")
            if weights_path and not Path(weights_path).exists():
                print(f"[WARN] Ear detector weights not found at: {weights_path}. Using center-crop fallback.")

    @staticmethod
    def _center_crop(img: PILImage.Image) -> PILImage.Image:
        w, h = img.size
        side = min(w, h)
        x1 = (w - side) // 2
        y1 = (h - side) // 2
        return img.crop((x1, y1, x1 + side, y1 + side))

    def _best_box(self, xyxy: np.ndarray, confs: np.ndarray) -> int:
        areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
        return int(np.argmax(areas * confs))

    @torch.no_grad()
    def detect_box(self, img: PILImage.Image) -> Optional[Tuple[int, int, int, int]]:
        if not self.enabled: return None
        arr = np.array(img)
        h, w = arr.shape[:2]
        try:
            r = self.model.predict(arr, conf=self.conf, iou=self.iou, verbose=False)[0]
            if r.boxes is None or r.boxes.xyxy is None or len(r.boxes) == 0: return None
            xyxy = r.boxes.xyxy.float().cpu().numpy()
            confs = r.boxes.conf.float().cpu().numpy()
            idx = self._best_box(xyxy, confs)
            x1, y1, x2, y2 = xyxy[idx]
            bw, bh = (x2 - x1), (y2 - y1)
            pad_px = self.pad * float(max(bw, bh))
            x1 -= pad_px; y1 -= pad_px; x2 += pad_px; y2 += pad_px
            if self.square:
                cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                side = max(x2 - x1, y2 - y1)
                x1, x2 = cx - side / 2.0, cx + side / 2.0
                y1, y2 = cy - side / 2.0, cy + side / 2.0
            x1 = int(np.floor(np.clip(x1, 0, w - 1)))
            y1 = int(np.floor(np.clip(y1, 0, h - 1)))
            x2 = int(np.ceil (np.clip(x2, 1, w)))
            y2 = int(np.ceil (np.clip(y2, 1, h)))
            if x2 <= x1 or y2 <= y1: return None
            return (x1, y1, x2, y2)
        except Exception:
            return None

    @torch.no_grad()
    def crop(self, img: PILImage.Image) -> PILImage.Image:
        box = self.detect_box(img)
        if box is None: return self._center_crop(img)
        x1, y1, x2, y2 = box
        return img.crop((x1, y1, x2, y2))



# ---------------------
# Dataset
# ---------------------
class EarVNDataset(Dataset):
    def __init__(self, items, label_map: Dict[str, int], cfg: dict, stage: str, detector: Optional[EarDetector]):
        self.items = items
        self.label_map = label_map
        self.cfg = cfg
        self.stage = stage
        self.detector = detector
        self.input_size = cfg["input_size"]
        self.crop_cache_dir = Path(cfg["crop_cache_dir"])
        ensure_dir(self.crop_cache_dir)
        self._build_transforms()

        # Debug sample saving state
        dbg = self.cfg.get("debug_samples", {})
        self._dbg_enabled = bool(dbg.get("enabled", False)) and (self.stage == "train")
        self._dbg_max = int(dbg.get("max_save", 48))
        self._dbg_saved = 0
        self._dbg_every = int(dbg.get("save_every_n_batches", 1))
        out_dir = Path(self.cfg.get("work_dir", ".")) / dbg.get("out_dir", "samples")
        self._dbg_dir = out_dir
        self._dbg_dir_rgb = out_dir / "rgb"
        self._dbg_dir_aug = out_dir / "aug"
        self._dbg_dir_contour = out_dir / "contour_rgb"
        for d in [self._dbg_dir_rgb, self._dbg_dir_aug, self._dbg_dir_contour]:
            try:
                d.mkdir(parents=True, exist_ok=True)
            except Exception:
                pass
    def _build_transforms(self):
        aug = self.cfg["aug"]
        if _HAS_ALB:
            t_list = []
            if self.stage == "train":
                size_pix = int(0.12 * self.input_size)
                try:
                    t_list.append(A.CoarseDropout(
                        max_holes=2, max_height=size_pix, max_width=size_pix,
                        min_holes=1, fill_value=0, p=0.25
                    ))
                except Exception:
                    try:
                        t_list.append(A.Cutout(
                            num_holes=2, max_h_size=size_pix, max_w_size=size_pix,
                            fill_value=0, p=0.25
                        ))
                    except Exception:
                        pass
                if aug.get("hflip", True): t_list.append(A.HorizontalFlip(p=0.5))
                if aug.get("vflip", False): t_list.append(A.VerticalFlip(p=0.1))
                if aug.get("rotate_deg", 0) > 0: t_list.append(A.Rotate(limit=aug["rotate_deg"], p=0.5, border_mode=0))
                if aug.get("brightness", 0) > 0 or aug.get("contrast", 0) > 0:
                    t_list.append(A.RandomBrightnessContrast(brightness_limit=aug["brightness"], contrast_limit=aug["contrast"], p=0.5))
                if aug.get("blur_prob", 0) > 0:
                    t_list.append(A.MotionBlur(p=aug["blur_prob"]))
                jpeg_p = aug.get("jpeg_prob", 0)
                if jpeg_p > 0:
                    try:
                        t_list.append(A.JpegCompression(quality_range=(60, 95), p=jpeg_p))
                    except AttributeError:
                        try:
                            t_list.append(A.ImageCompression(quality_range=(60, 95), p=jpeg_p))
                        except TypeError:
                            t_list.append(A.ImageCompression(quality_lower=60, quality_upper=95, p=jpeg_p))
                if aug.get("perspective_prob", 0) > 0:
                    t_list.append(A.Perspective(scale=(0.02, 0.05), p=aug["perspective_prob"]))
            t_list.extend([
                A.LongestMaxSize(max_size=self.input_size),
                A.PadIfNeeded(min_height=self.input_size,
                              min_width=self.input_size,
                              border_mode=0,
                              value=(0, 0, 0)),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ])
            self.transforms = ("alb", A.Compose(t_list))
        else:
            t_list = []
            if self.stage == "train":
                if aug.get("hflip", True): t_list.append(TVT.RandomHorizontalFlip(p=0.5))
                if aug.get("vflip", False): t_list.append(TVT.RandomVerticalFlip(p=0.1))
                if aug.get("rotate_deg", 0) > 0: t_list.append(TVT.RandomRotation(degrees=aug["rotate_deg"]))
                if aug.get("brightness", 0) > 0 or aug.get("contrast", 0) > 0:
                    t_list.append(TVT.ColorJitter(brightness=aug["brightness"], contrast=aug["contrast"]))
                if aug.get("perspective_prob", 0) > 0:
                    t_list.append(TVT.RandomPerspective(distortion_scale=0.3, p=aug["perspective_prob"]))
                if aug.get("blur_prob", 0) > 0:
                    t_list.append(TVT.GaussianBlur(kernel_size=3))
            t_list.extend([
                TVT.Resize((self.input_size, self.input_size)),
                TVT.ToTensor(),
                TVT.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ])
            self.transforms = ("tv", TVT.Compose(t_list))
            print("[WARN] Albumentations not found. Using torchvision transforms instead.")

    # inside EarVNDataset._crop_and_cache(self, img_path)
    def _crop_and_cache(self, img_path: Path) -> PILImage.Image:
        import numpy as np, cv2, tempfile, os
        rel = img_path.as_posix().replace("/", "__")
        cache_file = self.crop_cache_dir / f"{rel}.png"

        def _safe_read_pil(p):
            return PILImage.open(p).convert("RGB")

        def _safe_read_cv(p):
            data = np.fromfile(str(p), dtype=np.uint8)     # Windows-safe
            arr  = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if arr is None:
                raise RuntimeError("cv2.imdecode returned None")
            arr  = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
            return PILImage.fromarray(arr)

        # 1) Read (prefer cache)
        try:
            if cache_file.exists() and not self.cfg.get("force_recache", False):
                img = _safe_read_pil(cache_file)
            else:
                try:
                    img = _safe_read_pil(img_path)
                except Exception:
                    img = _safe_read_cv(img_path)

                # 2) (Optional) detector crop — sandboxed
                box = None
                if self.cfg.get("use_yolo_crop", True) and self.detector is not None and self.detector.enabled:
                    try:
                        box = self.detector.detect_box(img)
                    except Exception:
                        box = None

                if box is not None:
                    img = crop_with_box(img, box, pad_ratio=self.cfg["crop_square_pad"])
                else:
                    w, h = img.size
                    s = min(w, h); x1 = (w - s) // 2; y1 = (h - s) // 2
                    img = img.crop((x1, y1, x1 + s, y1 + s))

                # 3) Resize
                img = img.resize((self.input_size, self.input_size), PILImage.BILINEAR)

                # 4) Atomic cache write (prevents worker races)
                try:
                    cache_file.parent.mkdir(parents=True, exist_ok=True)
                    with tempfile.NamedTemporaryFile(delete=False, dir=str(cache_file.parent), suffix=".tmp") as tmp:
                        tmp_path = tmp.name
                    img.save(tmp_path)
                    os.replace(tmp_path, cache_file)   # atomic on Windows/NTFS
                except Exception:
                    # best-effort; ignore cache write failures
                    try:
                        if os.path.exists(tmp_path):
                            os.remove(tmp_path)
                    except Exception:
                        pass

        except Exception as e:
            raise RuntimeError(f"[CROP/CACHE] {img_path}: {e}") from e

        return img


    def __len__(self): return len(self.items)

    def __getitem__(self, idx):
        import numpy as np
        import cv2
        from pathlib import Path
        path, person_id = self.items[idx]
        label = self.label_map[person_id]

        # Load + crop once (PIL), and keep an RGB NumPy copy for debug
        img = self._crop_and_cache(path)  # PIL RGB
        arr_orig_for_debug = np.array(img)  # HWC RGB uint8

        kind, tfm = self.transforms

        if kind == "alb":
            # Albumentations expects NumPy HWC; call with named arg
            out = tfm(image=arr_orig_for_debug)
            x = out["image"]  # torch.Tensor (C,H,W), normalized

            # Build an augmented RGB for debug by inverse-normalizing the tensor
            # (Albumentations path uses ImageNet stats in your pipeline)
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            aug = x.permute(1, 2, 0).cpu().numpy()  # (H,W,C), float
            aug = np.clip(aug * std + mean, 0.0, 1.0)
            arr_aug_for_debug = (aug * 255.0).astype(np.uint8)

        else:
            # Torchvision branch: pass PIL directly; your TVT includes ToTensor+Normalize
            x = tfm(img)  # torch.Tensor (C,H,W), normalized

            # Inverse-normalize for debug visuals (ImageNet stats in your TVT)
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            aug = x.permute(1, 2, 0).cpu().numpy()
            aug = np.clip(aug * std + mean, 0.0, 1.0)
            arr_aug_for_debug = (aug * 255.0).astype(np.uint8)

        # ---- Debug samples (original, augmented, contour overlay) ----
        if getattr(self, "_dbg_enabled", False) and (self._dbg_saved < self._dbg_max):
            try:
                name = f"{Path(path).stem}_{idx:06d}.png"
                # save original RGB
                cv2.imwrite(str(self._dbg_dir_rgb / name),
                            cv2.cvtColor(arr_orig_for_debug, cv2.COLOR_RGB2BGR))
                # save augmented RGB (post-aug, approx. pre-normalization via inverse-norm)
                cv2.imwrite(str(self._dbg_dir_aug / name),
                            cv2.cvtColor(arr_aug_for_debug, cv2.COLOR_RGB2BGR))
                # simple contour overlay from Sobel magnitude
                gray = cv2.cvtColor(arr_aug_for_debug, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
                gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
                gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
                mag = np.sqrt(gx * gx + gy * gy)
                mag = (mag - mag.min()) / (mag.max() - mag.min() + 1e-8)
                heat8 = np.uint8(np.clip(mag, 0, 1) * 255)
                color = cv2.applyColorMap(heat8, cv2.COLORMAP_JET)
                overlay = np.clip(0.55 * cv2.cvtColor(arr_aug_for_debug, cv2.COLOR_RGB2BGR) / 255.0 +
                                0.45 * color / 255.0, 0, 1)
                cv2.imwrite(str(self._dbg_dir_contour / name), np.uint8(overlay * 255))
                self._dbg_saved += 1
            except Exception:
                pass

        return x, label


# ---------------------
# Lightweight Attention: CBAM + (store last maps)
# ---------------------
class ChannelAttention(nn.Module):
    def __init__(self, in_ch, reduction=16):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_ch, in_ch // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_ch // reduction, in_ch, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
        self.last_ca = None
    def forward(self, x):
        b, c, _, _ = x.size()
        avg = F.adaptive_avg_pool2d(x, 1).view(b, c)
        mx  = F.adaptive_max_pool2d(x, 1).view(b, c)
        out = self.mlp(avg) + self.mlp(mx)
        out = self.sigmoid(out).view(b, c, 1, 1)
        self.last_ca = out.detach()
        return x * out

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.last_sa = None
    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        s = torch.cat([avg, mx], dim=1)
        s = self.conv(s)
        s = self.sigmoid(s)
        self.last_sa = s.detach()
        return x * s

class CBAM(nn.Module):
    def __init__(self, in_ch, reduction=16, kernel_size=7):
        super().__init__()
        self.ca = ChannelAttention(in_ch, reduction)
        self.sa = SpatialAttention(kernel_size)
    def forward(self, x):
        return self.sa(self.ca(x))

# ---------------------
# ECA (for fusion)
# ---------------------
class ECA(nn.Module):
    def __init__(self, k_size=3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1)//2, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.last_weights = None
    def forward(self, x):
        y = self.avg_pool(x)                                    # (B,C,1,1)
        y1d = y.squeeze(-1).transpose(-1, -2)                   # (B,1,C)
        y1d = self.conv(y1d)
        y1d = self.sigmoid(y1d)                                 # (B,1,C)
        w = y1d.transpose(-1, -2).unsqueeze(-1)                 # (B,C,1,1)
        self.last_weights = w.detach()
        return x * w.expand_as(x)

def load_pretrained_if_exists(model: nn.Module, cfg: dict):
    """
    If cfg["pretrained_ckpt"] exists on disk, load backbone/embedding weights from it and skip class head.
    Logs [INFO] with missing/unexpected keys. Safe no-op if file absent.
    """
    try:
        ckpt_path = cfg.get("pretrained_ckpt", None)
        if not ckpt_path:
            print("[INFO] No pretrained_ckpt specified; skipping transfer.")
            return False
        from pathlib import Path as _P
        if not _P(ckpt_path).exists():
            print(f"[INFO] Pretrained checkpoint not found at {ckpt_path}; skipping transfer.")
            return False

        import torch as _torch
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        raw = ckpt.get("state_dict", ckpt.get("model", ckpt))
        # Flatten possible Lightning prefixes
        state = {}
        for k, v in raw.items():
            kk = k
            for pref in ("model.", "net.", "module."):
                if kk.startswith(pref):
                    kk = kk[len(pref):]
            state[kk] = v

        # Drop class-head parameters (ArcFace / classifier weights & any bias)
        drop_substrings = ("head.weight", "head.bias", "arcface", "classifier", "fc.weight", "fc.bias")
        state = {k: v for k, v in state.items() if not any(ds in k for ds in drop_substrings)}

        missing, unexpected = model.load_state_dict(state, strict=False)
        print("[INFO] Transfer loaded from", ckpt_path)
        print("[INFO] Missing keys:", len(missing))
        if len(missing) <= 20:
            print("[INFO]  ->", missing)
        else:
            print("[INFO]  -> (truncated) first 20:", missing[:20])
        print("[INFO] Unexpected keys:", len(unexpected))   
        if len(unexpected) <= 20:
            print("[INFO]  ->", unexpected)
        else:
            print("[INFO]  -> (truncated) first 20:", unexpected[:20])
        return True
    except Exception as e:
                print("[WARN] Failed to load pretrained checkpoint:", e)
    return False

def _cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    # a: [N,D], b: [M,D] (both L2-normalized already in your pipeline)
    return (a @ b.T).astype(np.float32)

def _compute_centroids(feats: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (centroids [C,D], unique_labels [C])"""
    u = np.unique(labels)
    cents = []
    for c in u:
        x = feats[labels == c]
        if len(x) == 0: continue
        m = x.mean(axis=0)
        m = m / (np.linalg.norm(m) + 1e-12)
        cents.append(m)
    if not cents:
        return np.zeros((0, feats.shape[1]), dtype=np.float32), np.array([], dtype=labels.dtype)
    return np.vstack(cents).astype(np.float32), u

def _eer_from_fpr_tpr(fpr, tpr):
    # Find point where FPR ~= 1-TPR (EER). Linear interpolation on ROC.
    fnr = 1 - tpr
    i = np.nanargmin(np.abs(fpr - fnr))
    return float((fpr[i] + fnr[i]) / 2.0)

def _tar_at_far(fpr, tpr, far_targets):
    """Return dict FAR->(TAR, thr) by interpolating ROC and thresholds from roc_curve()"""
    # We need mapping from thresholds to fpr/tpr; recompute to get thresholds
    # Caller should pass raw y_true and y_score to get thresholds alongside
    raise NotImplementedError  # Will be overridden by wrapper that knows thresholds

def _tar_at_far_with_thresholds(fpr, tpr, thresh, far_targets):
    # fpr,tpr,thresh from roc_curve(pos_label=1). Note: sklearn returns thresholds for scores descending.
    out = {}
    # Ensure monotonic by sorting FPR ascending
    order = np.argsort(fpr)
    fpr_s, tpr_s, thr_s = fpr[order], tpr[order], thresh[order]
    for far in far_targets:
        # index of closest FPR
        j = int(np.argmin(np.abs(fpr_s - float(far))))
        out[float(far)] = {"TAR": float(tpr_s[j]), "threshold": float(thr_s[j])}
    return out

def _plot_roc(fpr, tpr, out_png: Path, title="ROC"):
    fig = plt.figure(figsize=(5,5), dpi=150)
    plt.plot(fpr, tpr, lw=2)
    plt.plot([0,1],[0,1],'--', lw=1)
    plt.xlabel("FPR")
    plt.ylabel("TPR (TAR)")
    plt.title(title)
    plt.grid(True, ls="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)

def _plot_hist(genuine, impostor, out_png: Path, title="Score histogram"):
    fig = plt.figure(figsize=(6,4), dpi=150)
    plt.hist(genuine, bins=100, alpha=0.6, label="Genuine")
    plt.hist(impostor, bins=100, alpha=0.6, label="Impostor")
    plt.xlabel("Cosine similarity")
    plt.ylabel("Count")
    plt.title(title)
    plt.legend()
    plt.grid(True, ls="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)

def _plot_heatmap(M: np.ndarray, out_png: Path, title="Similarity"):
    fig = plt.figure(figsize=(6,5), dpi=150)
    plt.imshow(M, aspect="auto")
    plt.colorbar()
    plt.title(title)
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)

def save_full_verification_artifacts(
    test_feats: np.ndarray,          # (Nt, D) L2-normalized
    test_labels: np.ndarray,         # (Nt,)
    gallery_feats: np.ndarray | None,# (Ng, D) L2-normalized or None
    gallery_labels: np.ndarray | None,
    out_dir: str,
    cfg: dict
):
    """
    Builds:
      - test×test cosine matrix (+ heatmap)
      - test×centroid matrix (+ heatmap)  [gallery->centroids if given, else test->centroids]
      - pairs (sampled) + ROC + hist + AUC + EER + TAR@FAR w/ thresholds
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ---------- Matrices ----------
    # Test-Test
    S_tt = _cosine_sim(test_feats, test_feats)   # (Nt,Nt)
    np.save(out / "similarity_test_test.npy", S_tt)
    _plot_heatmap(S_tt, out / "similarity_test_test_heatmap.png", "Test × Test cosine")

    # Test-Centroid (gallery if available; else centroids from test)
    if gallery_feats is not None and gallery_labels is not None and len(gallery_feats):
        cents, cents_lbl = _compute_centroids(gallery_feats, gallery_labels)
        title = "Test × Gallery-Centroids cosine"
        stem = "test_centroid_from_gallery"
    else:
        cents, cents_lbl = _compute_centroids(test_feats, test_labels)
        title = "Test × Test-Centroids cosine"
        stem = "test_centroid_from_test"
    S_tc = _cosine_sim(test_feats, cents)  # (Nt, C)
    np.save(out / f"similarity_{stem}.npy", S_tc)
    _plot_heatmap(S_tc, out / f"similarity_{stem}_heatmap.png", title)

    # ---------- Pair sampling for ROC ----------
    vmax_g = int(cfg["verification"].get("max_genuine_pairs", 100000))
    vmax_i = int(cfg["verification"].get("max_impostor_pairs", 100000))
    rng = np.random.default_rng(12345)

    # Build index lists per class
    idx_by_cls = {c: np.where(test_labels == c)[0] for c in np.unique(test_labels)}
    # Genuine pairs
    gen_scores = []
    geni = []
    for c, idxs in idx_by_cls.items():
        if len(idxs) < 2: continue
        # all combinations or random subset if large
        pairs = list(itertools.combinations(idxs.tolist(), 2))
        rng.shuffle(pairs)
        for a, b in pairs[:max(0, vmax_g - len(gen_scores))]:
            s = float((test_feats[a] @ test_feats[b]))
            gen_scores.append(s); geni.append((int(a), int(b)))
            if len(gen_scores) >= vmax_g: break
        if len(gen_scores) >= vmax_g: break
    gen_scores = np.array(gen_scores, dtype=np.float32)

    # Impostor pairs
    all_idx = np.arange(len(test_labels))
    imp_scores = []
    impi = []
    # sample by drawing random pairs of different labels
    attempts = 0
    while len(imp_scores) < vmax_i and attempts < max(vmax_i*10, 100000):
        a = int(rng.integers(0, len(all_idx)))
        b = int(rng.integers(0, len(all_idx)))
        if a == b: 
            attempts += 1; continue
        if test_labels[a] == test_labels[b]:
            attempts += 1; continue
        s = float((test_feats[a] @ test_feats[b]))
        imp_scores.append(s); impi.append((a,b))
        attempts += 1
    imp_scores = np.array(imp_scores, dtype=np.float32)

    # Save pairs
    np.save(out / "pairs_genuine_idx.npy", np.array(geni, dtype=np.int32))
    np.save(out / "pairs_impostor_idx.npy", np.array(impi, dtype=np.int32))
    np.save(out / "pairs_genuine_scores.npy", gen_scores)
    np.save(out / "pairs_impostor_scores.npy", imp_scores)

    # ---------- ROC/AUC/EER/TAR ----------
    # Build labels & scores: 1 for genuine, 0 for impostor
    y = np.concatenate([np.ones_like(gen_scores, dtype=np.int32),
                        np.zeros_like(imp_scores, dtype=np.int32)])
    s = np.concatenate([gen_scores, imp_scores]).astype(np.float32)

    # sklearn ROC returns thresholds aligned to scores (desc)
    fpr, tpr, thr = roc_curve(y, s, pos_label=1)
    roc_auc = float(auc(fpr, tpr))
    eer = _eer_from_fpr_tpr(fpr, tpr)
    tar_far = _tar_at_far_with_thresholds(fpr, tpr, thr, cfg["verification"].get("far_targets", [1e-1, 5e-2, 1e-2, 1e-3]))

    # Plots
    _plot_roc(fpr, tpr, out / "roc_curve.png", title=f"ROC (AUC={roc_auc:.4f}, EER={eer:.4f})")
    _plot_hist(gen_scores, imp_scores, out / "score_histogram.png")

    # Summary
    summary = {
        "num_test": int(len(test_labels)),
        "num_classes_test": int(len(np.unique(test_labels))),
        "pairs": {
            "genuine": int(len(gen_scores)),
            "impostor": int(len(imp_scores))
        },
        "AUC": roc_auc,
        "EER": eer,
        "TAR_at_FAR": tar_far
    }
    with open(out / "verification_summary_full.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


# ---------------------
# Two-Stream ResNet50 (RGB vs. Contour) with ECA fusion @ stage3
# ---------------------
class TwoStreamResNet50(nn.Module):
    def __init__(self, embedding_dim=512, use_attention=True, attention_type="cbam", eca_k=3):
        super().__init__()
        mA = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        mB = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

        # Stream A (RGB)
        self.A_stem = nn.Sequential(mA.conv1, mA.bn1, mA.relu, mA.maxpool)
        self.A_l1, self.A_l2, self.A_l3, self.A_l4 = mA.layer1, mA.layer2, mA.layer3, mA.layer4

        # Stream B (Contour)
        self.B_stem = nn.Sequential(mB.conv1, mB.bn1, mB.relu, mB.maxpool)
        self.B_l1, self.B_l2, self.B_l3 = mB.layer1, mB.layer2, mB.layer3

        # Fusion after stage3
        self.eca = ECA(k_size=eca_k)
        self.fuse_reduce = nn.Conv2d(2048, 1024, kernel_size=1, bias=False)
        self.fuse_bn = nn.BatchNorm2d(1024)
        self.relu = nn.ReLU(inplace=True)

        # Head
        self.use_attention = use_attention
        self.attn = CBAM(2048, reduction=16, kernel_size=7) if use_attention else None
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.embed = nn.Linear(2048, embedding_dim)
        self.bn = nn.BatchNorm1d(embedding_dim)
        nn.init.xavier_normal_(self.embed.weight)
        nn.init.constant_(self.embed.bias, 0)
        nn.init.constant_(self.bn.weight, 1.0)
        nn.init.constant_(self.bn.bias, 0.0)

        # Buffers for contour generation
        k = torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]], dtype=torch.float32)
        self.register_buffer("_kx", k.view(1,1,3,3))
        self.register_buffer("_ky", k.t().contiguous().view(1,1,3,3))
        self.register_buffer("_lap", torch.tensor([[0,1,0],[1,-4,1],[0,1,0]], dtype=torch.float32).view(1,1,3,3))
        self.register_buffer("_mean", torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.register_buffer("_std",  torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1))

        # Expose "trunk" for freezing
        self.trunk = nn.ModuleList([
            self.A_stem, self.A_l1, self.A_l2, self.A_l3,
            self.B_stem, self.B_l1, self.B_l2, self.B_l3,
            self.eca, self.fuse_reduce, self.fuse_bn, self.A_l4
        ])

    @torch.no_grad()
    def _rgb_to_contour(self, x_rgb):
        # de-normalize to [0,1]-ish
        xr = x_rgb * self._std + self._mean
        r, g, b = xr[:,0:1], xr[:,1:2], xr[:,2:3]

        # Remove bright green overlays (debug rectangles)
        mask_green = (g > 0.6) & (g > r + 0.25) & (g > b + 0.25)
        if mask_green.any():
            avg = F.avg_pool2d(xr, kernel_size=3, stride=1, padding=1)
            xr = torch.where(mask_green.repeat(1,3,1,1), avg, xr)
            r, g, b = xr[:,0:1], xr[:,1:2], xr[:,2:3]

        gray = 0.2989*r + 0.5870*g + 0.1140*b

        # Low-variance mask (uniform occluders)
        kvar = 9
        mean = F.avg_pool2d(gray, kernel_size=kvar, stride=1, padding=kvar//2)
        mean2 = F.avg_pool2d(gray*gray, kernel_size=kvar, stride=1, padding=kvar//2)
        var = torch.clamp(mean2 - mean*mean, min=0.0)
        var_thresh = 1e-4
        mask_lowvar = (var < var_thresh).float()
        mask_dil = F.max_pool2d(mask_lowvar, kernel_size=5, stride=1, padding=2)
        mask_erode = 1.0 - F.max_pool2d(1.0 - mask_dil, kernel_size=3, stride=1, padding=1)
        suppress = torch.clamp(1.0 - mask_erode, 0.0, 1.0)

        gx = F.conv2d(gray, self._kx, padding=1) * suppress
        gy = F.conv2d(gray, self._ky, padding=1) * suppress
        lap = F.conv2d(gray, self._lap, padding=1) * suppress

        def _mm(t):
            mn = t.amin(dim=(2,3), keepdim=True)
            mx = t.amax(dim=(2,3), keepdim=True)
            return (t - mn) / (mx - mn + 1e-8)
        return torch.cat([_mm(gx), _mm(gy), _mm(lap)], dim=1), suppress

    def forward(self, x, return_aux: bool=False):
        x_rgb = x[:, :3, :, :] if x.size(1) >= 3 else x.repeat(1,3,1,1)

        # A-stream
        xa = self.A_stem(x_rgb)
        xa = self.A_l1(xa); xa = self.A_l2(xa); xa = self.A_l3(xa)   # (B,1024,*,*)

        # B-stream (from RGB)
        x_cont, suppress = self._rgb_to_contour(x_rgb)
        xb = self.B_stem(x_cont)
        xb = self.B_l1(xb); xb = self.B_l2(xb); xb = self.B_l3(xb)    # (B,1024,*,*)

        # Fuse @ stage3
        x3 = torch.cat([xa, xb], dim=1)                               # (B,2048,*,*)
        x3 = self.eca(x3)
        x3 = self.relu(self.fuse_bn(self.fuse_reduce(x3)))            # (B,1024,*,*)

        # Shared layer4
        x4 = self.A_l4(x3)                                            # (B,2048,*,*)
        if self.use_attention and self.attn is not None:
            x4 = self.attn(x4)
        xg = self.gap(x4).flatten(1)
        emb = self.bn(self.embed(xg))
        emb = F.normalize(emb, dim=1)

        if return_aux:
            aux = {
                "contour": x_cont.detach(),     # (B,3,H,W) normalized channels
                "suppress": suppress.detach(),  # (B,1,H,W)
                "fused3": x3.detach(),          # (B,1024,h,w)
                "x4": x4.detach(),
                "eca_w": self.eca.last_weights.detach() if self.eca.last_weights is not None else None,
                "cbam_sa": self.attn.sa.last_sa if (self.use_attention and self.attn is not None) else None,
                "cbam_ca": self.attn.ca.last_ca if (self.use_attention and self.attn is not None) else None
            }
            return emb, aux
        return emb

# ---------------------
# Edge/Contour helpers
# ---------------------
def _canny_edge(arr_rgb_uint8: np.ndarray, th1=60, th2=160, aperture=3) -> np.ndarray:
    g = cv2.cvtColor(arr_rgb_uint8, cv2.COLOR_RGB2GRAY)
    e = cv2.Canny(g, th1, th2, apertureSize=aperture, L2gradient=True)
    return (e.astype(np.float32) / 255.0)

def _sobel_edge(arr_rgb_uint8: np.ndarray) -> np.ndarray:
    g = cv2.cvtColor(arr_rgb_uint8, cv2.COLOR_RGB2GRAY)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    m = np.sqrt(gx*gx + gy*gy)
    m = m / (m.max() + 1e-6)
    return m.astype(np.float32)

def stack_rgb_edge_to_tensor(arr_rgb_uint8: np.ndarray, cfg) -> torch.Tensor:
    """Return 4xHxW tensor (RGB normalized + EDGE normalized)."""
    H, W, _ = arr_rgb_uint8.shape
    if cfg["edge"]["method"].lower() == "sobel":
        edge = _sobel_edge(arr_rgb_uint8)
    else:
        edge = _canny_edge(arr_rgb_uint8, cfg["edge"]["canny_th1"], cfg["edge"]["canny_th2"], cfg["edge"]["aperture"])
    rgb = arr_rgb_uint8.astype(np.float32) / 255.0
    x = np.dstack([rgb, edge[..., None]])  # H W 4
    x = torch.from_numpy(x).permute(2,0,1)  # 4 H W
    # normalize
    mean = torch.tensor(cfg["norm"]["rgb_mean"] + [cfg["norm"]["edge_mean"]]).view(4,1,1)
    std  = torch.tensor(cfg["norm"]["rgb_std"]  + [cfg["norm"]["edge_std"]]).view(4,1,1)
    x = (x - mean) / std
    return x.float()

# ---------------------
# Legacy single-stream backbone retained (for completeness)
# ---------------------
class BackboneResNet50(nn.Module):
    def __init__(self, embedding_dim=512, use_attention=True, attention_type="cbam"):
        super().__init__()
        m = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.stem = nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool)
        self.layer1, self.layer2, self.layer3, self.layer4 = m.layer1, m.layer2, m.layer3, m.layer4
        self.trunk = nn.ModuleList([self.stem, self.layer1, self.layer2, self.layer3, self.layer4])
        self.use_attention = use_attention
        self.attn = CBAM(2048, reduction=16, kernel_size=7) if use_attention else None
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.embed = nn.Linear(2048, embedding_dim)
        self.bn = nn.BatchNorm1d(embedding_dim)
        nn.init.xavier_normal_(self.embed.weight)
        nn.init.constant_(self.embed.bias, 0)
        nn.init.constant_(self.bn.weight, 1.0)
        nn.init.constant_(self.bn.bias, 0.0)
    def forward(self, x, return_aux: bool=False):
        x = self.stem(x); x = self.layer1(x); x = self.layer2(x); x = self.layer3(x); x4 = self.layer4(x)
        if self.use_attention and self.attn is not None: x4 = self.attn(x4)
        xg = self.gap(x4).flatten(1)
        emb = self.bn(self.embed(xg)); emb = F.normalize(emb, dim=1)
        if return_aux:
            aux = {
                "contour": None, "suppress": None, "fused3": None, "x4": x4.detach(),
                "eca_w": None,
                "cbam_sa": self.attn.sa.last_sa if (self.use_attention and self.attn is not None) else None,
                "cbam_ca": self.attn.ca.last_ca if (self.use_attention and self.attn is not None) else None
            }
            return emb, aux
        return emb

# ---------------------
# EDH Head (auxiliary) and ONSA adversary
# ---------------------
class EDHHead(nn.Module):
    """
    Predicts a (downsampled) distance map from fused Stage-3 features.
    Training is self-supervised against a pseudo Signed/Unsigned Distance Transform (SDT) from edges.
    """
    def __init__(self, in_ch=1024, out_stride=4):
        super().__init__()
        ch = max(64, in_ch // 16)
        self.proj = nn.Sequential(
            nn.Conv2d(in_ch, ch, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(ch, 1, 1)
        )
        self.out_stride = out_stride

    def forward(self, x3):
        # x3: (B,1024,h,w), predict at h/out_stride, w/out_stride -> we'll downsample x3 internally
        if self.out_stride > 1:
            x3 = F.avg_pool2d(x3, kernel_size=self.out_stride, stride=self.out_stride)
        d = self.proj(x3)  # (B,1,h',w')
        return d

def eikonal_loss(d):
    # |∇d| -> target 1
    gx = F.conv2d(d, torch.tensor([[[[-1,0,1],[-2,0,2],[-1,0,1]]]], dtype=d.dtype, device=d.device), padding=1)
    gy = F.conv2d(d, torch.tensor([[[[-1,-2,-1],[0,0,0],[1,2,1]]]], dtype=d.dtype, device=d.device), padding=1)
    grad_mag = torch.sqrt(gx*gx + gy*gy + 1e-8)
    return F.l1_loss(grad_mag, torch.ones_like(grad_mag))

class OccluderEncoder(nn.Module):
    """
    Encodes occluder-focused pooled features into a direction u in embedding space.
    """
    def __init__(self, in_ch=1024, emb_dim=512):
        super().__init__()
        hid = max(128, in_ch // 4)
        self.net = nn.Sequential(
            nn.Linear(in_ch, hid), nn.ReLU(inplace=True),
            nn.Linear(hid, emb_dim)
        )
    def forward(self, v):
        u = self.net(v)                           # (B,emb)
        u = F.normalize(u, dim=1)
        return u

# ---------------------
# ArcFace components
# ---------------------
class ArcMarginProduct(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.50, easy_margin=False):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.s = s; self.m = m; self.easy_margin = easy_margin
        self.cos_m = math.cos(m); self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m); self.mm = math.sin(math.pi - m) * m
    # inside class ArcMarginProduct(nn.Module)
    def forward(self, embeddings, labels, m: Optional[float] = None, s: Optional[float] = None):
        """
        ArcFace forward with optional per-call overrides:
        - m: margin override (else use self.m)
        - s: scale  override (else use self.s)
        """
        # effective margin/scale
        m_eff = float(self.m if m is None else m)
        s_eff = float(self.s if s is None else s)

        # on-the-fly trigs for current margin
        cos_m = math.cos(m_eff); sin_m = math.sin(m_eff)
        th    = math.cos(math.pi - m_eff)
        mm    = math.sin(math.pi - m_eff) * m_eff

        cosine = F.linear(embeddings, F.normalize(self.weight))
        sine   = torch.sqrt(1.0 - torch.clamp(cosine**2, 0, 1))
        phi    = cosine * cos_m - sine * sin_m
        if not self.easy_margin:
            phi = torch.where(cosine > th, phi, cosine - mm)
        else:
            phi = torch.where(cosine > 0,  phi,  cosine)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1,1), 1.0)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        return output * s_eff


# ---------------------
# Classifier wrapper with optional EDH/ONSA
# ---------------------
class ArcFaceClassifier(nn.Module):
    def __init__(self, num_classes, embedding_dim=512, s=30.0, m=0.50,
                 use_attention=True, attention_type="cbam",
                 backbone_type="two_stream", eca_k=3,
                 edh_cfg=None, onsa_cfg=None):
        super().__init__()
        self.backbone_type = backbone_type
        if backbone_type == "two_stream":
            self.backbone = TwoStreamResNet50(embedding_dim=embedding_dim, use_attention=use_attention,
                                              attention_type=attention_type, eca_k=eca_k)
            edh_in = 1024
        else:
            self.backbone = BackboneResNet50(embedding_dim=embedding_dim, use_attention=use_attention,
                                             attention_type=attention_type)
            edh_in = 1024  # unused in this case

        self.head = ArcMarginProduct(embedding_dim, num_classes, s=s, m=m)

        # EDH
        self.edh_enabled = bool(edh_cfg and edh_cfg.get("enabled", False))
        self.edh_l1_w = 0.0; self.edh_eik_w = 0.0; self.edh_down = 4
        if self.edh_enabled and backbone_type == "two_stream":
            self.edh_l1_w = float(edh_cfg.get("l1_weight", 0.2))
            self.edh_eik_w = float(edh_cfg.get("eik_weight", 0.1))
            self.edh_down = int(edh_cfg.get("downsample", 4))
            self.edh_head = EDHHead(in_ch=edh_in, out_stride=self.edh_down)
        else:
            self.edh_head = None

        # ONSA
        self.onsa_enabled = bool(onsa_cfg and onsa_cfg.get("enabled", False))
        self.onsa_w = float(onsa_cfg.get("weight", 0.05)) if self.onsa_enabled else 0.0
        self.onsa_adv_w = float(onsa_cfg.get("adv_weight", 0.05)) if self.onsa_enabled else 0.0
        if self.onsa_enabled and backbone_type == "two_stream":
            self.occ_encoder = OccluderEncoder(in_ch=edh_in, emb_dim=embedding_dim)
        else:
            self.occ_encoder = None

    def forward_with_aux(self, x, labels=None, current_m: Optional[float] = None, current_s: Optional[float] = None):
        emb, aux = self.backbone(x, return_aux=True)
        logits = None if labels is None else self.head(emb, labels, current_m, current_s)
        extra_losses = {}

        # EDH loss (stop-grad into backbone)
        if self.edh_enabled and self.edh_head is not None and aux.get("fused3") is not None and aux.get("contour") is not None:
            with torch.no_grad():
                # Pseudo-distance from contour magnitude
                cont = aux["contour"]                      # (B,3,H,W) in [0,1] approx
                # binary edge map from |SobelX|+|SobelY|+|Lap|
                mag = torch.mean(cont.abs(), dim=1, keepdim=True)  # (B,1,H,W)
                # normalize and threshold softly
                mag_n = (mag - mag.amin((2,3),keepdim=True)) / (mag.amax((2,3),keepdim=True) - mag.amin((2,3),keepdim=True) + 1e-8)
                edge_bin = (mag_n > 0.25).float()
                inv = 1.0 - torch.clamp(edge_bin, 0, 1)
                # OpenCV distance transform per-sample
                B, _, H, W = inv.shape
                sdt_list = []
                for b in range(B):
                    msk = (inv[b,0].cpu().numpy() * 255.0).astype(np.uint8)
                    dt = cv2.distanceTransform(msk, distanceType=cv2.DIST_L2, maskSize=3)
                    if dt.max() > 0: dt = dt / (dt.max() + 1e-8)
                    sdt_list.append(torch.from_numpy(dt).to(inv.device).float().unsqueeze(0))
                d_gt = torch.stack(sdt_list, dim=0)  # (B,1,H,W)
                # match EDH resolution
                if self.edh_down > 1:
                    d_gt = F.avg_pool2d(d_gt, kernel_size=self.edh_down, stride=self.edh_down)

            with torch.no_grad():
                x3_detached = aux["fused3"].detach()
            d_pred = self.edh_head(x3_detached)   # (B,1,h',w')
            # --- match target to prediction spatially ---
            if d_gt.shape[-2:] != d_pred.shape[-2:]:
                d_gt = F.interpolate(d_gt, size=d_pred.shape[-2:], mode="area")
            l1 = F.l1_loss(torch.sigmoid(d_pred), d_gt)
            ek = eikonal_loss(d_pred)
            extra_losses["edh_l1"] = l1 * self.edh_l1_w
            extra_losses["edh_eik"] = ek * self.edh_eik_w


        # ONSA
        if self.onsa_enabled and self.occ_encoder is not None and aux.get("fused3") is not None and aux.get("suppress") is not None:
            x3 = aux["fused3"]          # (B,1024,h,w)
            sup = aux["suppress"]       # (B,1,H,W) at input res
            # bring suppress to fused3 size
            sup_r = F.interpolate(sup, size=x3.shape[-2:], mode="nearest")
            # masked global average to get occluder-focused vector
            wsum = sup_r.sum(dim=(2,3), keepdim=True) + 1e-8
            v_occ = (x3 * (1.0 - sup_r)).sum(dim=(2,3)) / ((1.0 - sup_r).sum(dim=(2,3)) + 1e-8)  # prefer occluder region
            # Adversary tries to ALIGN u with current emb (maximize dot^2), backbone penalizes alignment
            u = self.occ_encoder(v_occ.detach())                   # adversary sees fixed features -> learns to align with z
            dot = torch.sum(emb * u.detach(), dim=1)               # penalize backbone w.r.t fixed u
            onsa_pen = torch.mean(dot * dot) * self.onsa_w
            # adversary objective (maximize dot^2): implement as negative loss to update occ_encoder
            dot_adv = torch.sum(emb.detach() * u, dim=1)
            onsa_adv = - torch.mean(dot_adv * dot_adv) * self.onsa_adv_w
            extra_losses["onsa_pen"] = onsa_pen
            extra_losses["onsa_adv"] = onsa_adv

        return emb, logits, aux, extra_losses

    def forward(self, x, labels=None):
        emb = self.backbone(x, return_aux=False)
        logits = None if labels is None else self.head(emb, labels)
        return emb, logits


# ---------------------
# Splits
# ---------------------
def split_dataset(cfg) -> Tuple[List[Tuple[Path,str]], List[Tuple[Path,str]], List[Tuple[Path,str]], Dict[str,int]]:
    """
    Supports:
      • UERC-2019 style: base/PartA/<pid>/* , base/PartB/<pid>/*
      • Flat AMI style:  <base>/*.jpg where filenames are like "000_back_ear.jpg"
                         (all images that start with the same 3-digit prefix belong to the same person)
    """
    base = Path(cfg["dataset_dir"])
    rng = random.Random(cfg["seed"])

    def _collect_person_dirs_uerc(base: Path):
        partA = base / "PartA"
        partB = base / "PartB"
        pairs = []
        if partA.exists() or partB.exists():
            if partA.exists():
                for d in sorted(p for p in partA.iterdir() if p.is_dir()):
                    pairs.append((f"A_{d.name}", d))
            if partB.exists():
                for d in sorted(p for p in partB.iterdir() if p.is_dir()):
                    pairs.append((f"B_{d.name}", d))
            return pairs
        # fallback: assume flat person folders under base (EarVN10-style)
        return [(d.name, d) for d in sorted(base.iterdir()) if d.is_dir()]

    # Helper: enumerate images directly under base (flat layout)
    def _list_base_images(base: Path):
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
        return [p for p in sorted(base.glob("*")) if p.is_file() and p.suffix.lower() in exts]

    # --- Detect AMI flat layout first (files directly in base) ---
    base_images = _list_base_images(base)
    items_by_person: Dict[str, List[Path]] = {}

    if len(base_images) > 0:
        # AMI style: filenames like "000_back_ear.jpg" → person_id = "000"
        # We'll group by the leading 3-digit prefix before the first underscore.
        for p in base_images:
            stem = p.stem  # e.g., "000_back_ear"
            # Extract leading 3 or 4 digits; fallback to digits before first underscore
            m = re.match(r"^(\d{3,4})[_\-\.]", stem)
            if m:
                pid_core = m.group(1)
            else:
                # As a last resort, take consecutive leading digits
                m2 = re.match(r"^(\d+)", stem)
                if not m2:
                    # skip files that don't start with a person id
                    continue
                pid_core = m2.group(1)
                # zero-pad to length 3 if short (e.g., "7" → "007")
                if len(pid_core) < 3:
                    pid_core = pid_core.zfill(3)
            pid = pid_core  # keep clean label (no A_/B_ prefix for AMI)
            items_by_person.setdefault(pid, []).append(p)
    else:
        # UERC / folder-based layout
        for pid, pd in _collect_person_dirs_uerc(base):
            imgs = list_images(pd)
            if len(imgs) < cfg["min_images_person"]:
                continue
            items_by_person[pid] = imgs

    # Filter by minimum images per person
    min_imgs = int(cfg.get("min_images_person", 1))
    items_by_person = {pid: imgs for pid, imgs in items_by_person.items() if len(imgs) >= min_imgs}

    persons = sorted(items_by_person.keys())
    label_map = {pid: i for i, pid in enumerate(persons)}

    train_items, val_items, test_items = [], [], []
    cap = int(cfg.get("max_images_per_person", 10**9))
    for pid in persons:
        imgs = items_by_person[pid][:]  # copy
        rng.shuffle(imgs)
        n = len(imgs)
        # compute split sizes
        n_tr = int(round(cfg["train_frac"] * n))
        n_val = int(round(cfg["val_frac"] * n))
        # ensure at least 1 test sample when possible
        if n >= 3:
            n_tr = max(1, min(n_tr, n - 2))
            n_val = max(0, min(n_val, n - n_tr - 1))
        elif n == 2:
            n_tr, n_val = 1, 0
        else:  # n == 1 already filtered out by min_images_person
            n_tr, n_val = 1, 0

        tr = imgs[:n_tr]
        va = imgs[n_tr:n_tr + n_val]
        te = imgs[n_tr + n_val:]

        # cap per split
        tr = tr[:cap]; va = va[:cap]; te = te[:cap]

        train_items += [(p, pid) for p in tr]
        val_items   += [(p, pid) for p in va]
        test_items  += [(p, pid) for p in te]
        

        from collections import defaultdict
    import numpy as np

    def _by_pid(items):
        d = defaultdict(list)
        for pth, pid in items:
            d[pid].append((pth, pid))
        return d

    train_by = _by_pid(train_items)
    val_by   = _by_pid(val_items)
    test_by  = _by_pid(test_items)

    merged_test = []
    # FIX: ensure all are sets, not lists
    all_pids = sorted(set(train_by.keys()) | set(val_by.keys()) | set(test_by.keys()))
    rng = np.random.default_rng(getattr(cfg, "seed", 12345))

    for pid in all_pids:
        v = val_by.get(pid, [])
        t = test_by.get(pid, [])

        pick_val = v[0:1]        # one val if available
        pick_test = t[0:1]       # one test if available

        if len(pick_val) == 0 and len(t) > 0:
            pick_val = t[0:1]
            t = t[1:]

        if len(pick_test) == 0:
            if len(t) > 0:
                pick_test = t[0:1]
            elif len(v) > 1:
                pick_test = v[1:2]

        picks = pick_val + pick_test
        if len(picks) < 2:
            pool = v[1:] + t
            rng.shuffle(pool)
            for it in pool:
                if it not in picks:
                    picks.append(it)
                    if len(picks) == 2:
                        break

        merged_test += picks

    test_items = merged_test

    n_persons = len(all_pids)
    return train_items, val_items, test_items, label_map

# ---------------------
# Metrics / Eval helpers
# ---------------------
def accuracy_topk(logits, labels, ks=(1,)):
    with torch.no_grad():
        maxk = max(ks)
        _, pred = logits.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(labels.view(1, -1).expand_as(pred))
        res = []
        for k in ks:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append((correct_k.item() * (100.0 / labels.size(0))))
        return res

@torch.no_grad()
def extract_embeddings(model, dl, device):
    model.eval()
    embs, labels = [], []
    for imgs, y in dl:
        imgs = imgs.to(device)
        e, _ = model(imgs, None)
        embs.append(e.cpu().numpy()); labels.append(y.numpy())
    return np.concatenate(embs), np.concatenate(labels)

# ---------------------
# Open-set & Calibration Utils (from v2, minimal edits)
# ---------------------
@torch.no_grad()
def collect_logits(model, dl, device):
    """
    Run a dataloader and collect raw logits (pre-softmax), labels, and MSP at T=1.
    Compatible with Q-ArcFace head. Uses normalized features and class weights.
    """
    model.eval()
    all_logits, all_labels = [], []
    for imgs, labels in dl:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # forward to get normalized embedding
        feats = model.backbone(imgs)        # (B, d), already L2-normalized in your v3
        # compute class logits using normalized weights (ArcFace style)
        W = model.head.weight               # shape (C*K or C, d)
        W = torch.nn.functional.normalize(W, dim=1)
        logits = torch.matmul(feats, W.t())  # (B, C) if SubCenter aggregated; else adapt if needed

        all_logits.append(logits.detach().cpu())
        all_labels.append(labels.detach().cpu())

    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    msp = torch.softmax(all_logits, dim=1).max(dim=1).values
    return all_logits, all_labels, msp


def fit_temperature(val_logits, val_labels):
    """
    Find temperature T>0 to minimize NLL on validation labels (Platt-like scaling).
    Simple 1D search for robustness (no extra deps).
    """
    import numpy as np
    import torch.nn.functional as F

    Ts = np.linspace(0.5, 5.0, 46)  # 0.5 .. 5.0 step 0.1
    best_T, best_nll = 1.0, float("inf")
    y = val_labels.long()
    for T in Ts:
        sm = F.log_softmax(val_logits / T, dim=1)
        nll = -sm[torch.arange(len(y)), y].mean().item()
        if nll < best_nll:
            best_nll, best_T = nll, float(T)
    return best_T

@torch.no_grad()
def calibrate_and_apply_open_set(model, train_embs, train_labels, val_dl, test_dl, out_dir, cfg):
    """
    1) Temperature-calibrate on val logits.
    2) Choose MSP threshold (percentile) using correctly classified val samples.
    3) Build train centroids; choose cosine threshold using correctly classified val samples.
    4) Apply both rules on test; save confusion matrices with an 'unknown' bin.

    Robust to CUDA tensors, non-contiguous labels, and empty-correct sets.
    """
    import json
    from pathlib import Path
    import numpy as np
    import torch
    import torch.nn.functional as F
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix

    if not cfg.get("open_set", {}).get("enabled", True):
        return

    os_dir = Path(out_dir) / "open_set"
    os_dir.mkdir(parents=True, exist_ok=True)

    device = cfg["device"]

    # ---------------- helpers ----------------
    def _to_cpu_np(t):
        if isinstance(t, torch.Tensor):
            return t.detach().cpu().numpy()
        return np.asarray(t)

    def _softmax(logits, T=1.0):
        return torch.softmax(logits / float(T), dim=1)

    def _ensure_contiguous_labels(y):
        """Map arbitrary integer labels to 0..C-1 and return (mapped, classes_sorted, map_dict)."""
        y = np.asarray(y, dtype=int)
        classes_sorted = np.unique(y)
        idx_map = {int(c): i for i, c in enumerate(classes_sorted)}
        y_mapped = np.vectorize(idx_map.__getitem__)(y)
        return y_mapped, classes_sorted, idx_map

    # ---------------- 1) temperature calibration on val ----------------
    val_logits, val_labels, _ = collect_logits(model, val_dl, device)  # logits: torch.Tensor [N,C], labels: torch.LongTensor
    T = 1.0
    if cfg["open_set"].get("temperature_calibration", True):
        T = float(fit_temperature(val_logits, val_labels))  # make sure it's a float
    with open(os_dir / "calibration.json", "w") as f:
        json.dump({"temperature": T}, f, indent=2)

    # MSP threshold on correctly-classified val
    sm_val = _softmax(val_logits, T=T)                      # torch.Tensor
    pred_val = sm_val.argmax(dim=1)                         # torch.LongTensor
    correct_mask = (pred_val == val_labels)
    correct_msp = sm_val.max(dim=1).values[correct_mask]    # torch.Tensor (possibly empty)
    perc = int(cfg["open_set"].get("val_threshold_percentile", 10))

    if correct_msp.numel() > 0:
        # torch quantile -> scalar
        msp_tau = torch.quantile(correct_msp, q=min(max(perc/100.0, 0.0), 1.0)).item()
    else:
        # conservative fallback
        msp_tau = 0.5

    with open(os_dir / "msp_threshold.json", "w") as f:
        json.dump({"percentile": perc, "threshold": float(msp_tau)}, f, indent=2)

    # ---------------- 2) cosine-to-centroid threshold from TRAIN ----------------
    trX = np.asarray(train_embs, dtype=np.float32)
    trY = np.asarray(train_labels, dtype=int)
    trY_mapped, classes_sorted, idx_map = _ensure_contiguous_labels(trY)  # map to 0..C-1

    # centroids in embedding space
    C = []
    for c in range(len(classes_sorted)):
        C.append(trX[trY_mapped == c].mean(axis=0))
    C = np.stack(C, axis=0).astype(np.float32)
    C /= (np.linalg.norm(C, axis=1, keepdims=True) + 1e-8)  # L2-normalize centroids

    # val embeddings & max centroid similarity (use same val_dl)
    val_embs, val_lbls = extract_embeddings(model, val_dl, device)  # np or torch
    val_embs = _to_cpu_np(val_embs).astype(np.float32)
    val_lbls = _to_cpu_np(val_lbls).astype(int)
    val_lbls_mapped = np.vectorize(idx_map.get)(val_lbls)  # may yield None if label not in train

    # mask out samples whose labels are unknown to train (None)
    known_mask = np.array([l is not None for l in val_lbls_mapped], dtype=bool)
    if not known_mask.any():
        # if nothing is known, fallback conservative threshold
        cos_tau = 0.0
    else:
        val_lbls_mapped = np.asarray([int(l) if l is not None else -1 for l in val_lbls_mapped], dtype=int)
        val_embs /= (np.linalg.norm(val_embs, axis=1, keepdims=True) + 1e-8)
        sims_val = val_embs @ C.T  # [N_val, C]
        pred_c = sims_val.argmax(axis=1)
        max_sim = sims_val.max(axis=1)
        correct_mask_sim = (pred_c == val_lbls_mapped) & known_mask
        perc_cos = int(cfg["open_set"].get("cosine_threshold_percentile", 5))
        if np.any(correct_mask_sim):
            cos_tau = float(np.quantile(max_sim[correct_mask_sim], q=min(max(perc_cos/100.0, 0.0), 1.0)))
        else:
            cos_tau = 0.0  # fallback conservative
    with open(os_dir / "cosine_threshold.json", "w") as f:
        json.dump({"percentile": int(cfg["open_set"].get("cosine_threshold_percentile", 5)),
                   "threshold": float(cos_tau)}, f, indent=2)

    # ---------------- 3) apply to TEST: MSP rule ----------------
    test_logits, test_labels, _ = collect_logits(model, test_dl, device)
    sm_test = _softmax(test_logits, T=T)
    msp_test = sm_test.max(dim=1).values
    pred_test = sm_test.argmax(dim=1)

    # Map test labels to contiguous indices (unknown labels set to -1 for matrix)
    test_labels_np = _to_cpu_np(test_labels).astype(int)
    test_labels_mapped = np.vectorize(idx_map.get)(test_labels_np)
    test_labels_mapped = np.asarray([(-1 if l is None else int(l)) for l in test_labels_mapped], dtype=int)

    unknown = (msp_test < msp_tau)
    y_pred_os = _to_cpu_np(pred_test).astype(int)
    y_pred_os[unknown.cpu().numpy()] = -1  # -1 is "unknown"

    # Confusion (MSP)
    # labels for confusion matrix: 0..C-1 plus 'unknown' as last index C
    Cn = len(classes_sorted)
    y_true_cm = np.where(test_labels_mapped < 0, Cn, test_labels_mapped)
    y_pred_cm = np.where(y_pred_os < 0, Cn, y_pred_os)
    cm = confusion_matrix(y_true_cm, y_pred_cm, labels=list(range(Cn + 1)))

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest')
    plt.title("Open-set Confusion (MSP)")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(os_dir / "confusion_open_set_msp.png", dpi=160)
    plt.close()

    # ---------------- 4) apply to TEST: cosine-to-centroid rule ----------------
    test_embs, test_lbls_ = extract_embeddings(model, test_dl, device)
    test_embs = _to_cpu_np(test_embs).astype(np.float32)
    test_lbls_ = _to_cpu_np(test_lbls_).astype(int)
    test_lbls_mapped = np.vectorize(idx_map.get)(test_lbls_)
    test_lbls_mapped = np.asarray([(-1 if l is None else int(l)) for l in test_lbls_mapped], dtype=int)

    test_embs_n = test_embs / (np.linalg.norm(test_embs, axis=1, keepdims=True) + 1e-8)
    sims_test = test_embs_n @ C.T
    pred_c_test = sims_test.argmax(axis=1).astype(int)
    max_sim_test = sims_test.max(axis=1)
    unknown_cos = (max_sim_test < cos_tau)
    y_pred_cos = pred_c_test.copy()
    y_pred_cos[unknown_cos] = -1

    y_true_cm2 = np.where(test_lbls_mapped < 0, Cn, test_lbls_mapped)
    y_pred_cm2 = np.where(y_pred_cos < 0, Cn, y_pred_cos)
    cm2 = confusion_matrix(y_true_cm2, y_pred_cm2, labels=list(range(Cn + 1)))

    plt.figure(figsize=(8, 6))
    plt.imshow(cm2, interpolation='nearest')
    plt.title("Open-set Confusion (Cosine)")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(os_dir / "confusion_open_set_cosine.png", dpi=160)
    plt.close()


def plot_curves(train_losses, val_losses, train_accs, val_accs, out_png):
    plt.figure(figsize=(7,5))
    plt.title("Training & Validation")
    xs = np.arange(1, len(train_losses)+1)
    plt.plot(xs, train_losses, label="train_loss")
    plt.plot(xs, val_losses, label="val_loss")
    plt.plot(xs, train_accs, label="train_acc1")
    plt.plot(xs, val_accs, label="val_acc1")
    plt.xlabel("Epoch")
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(out_png, dpi=160); plt.close()

def save_confusion_matrix_png(y_true, y_pred, labels_names, out_png, max_classes_to_show=50):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(labels_names))))
    C = cm
    labels_show = labels_names[:max_classes_to_show] if len(labels_names) > max_classes_to_show else labels_names
    if len(labels_names) > max_classes_to_show:
        C = cm[:max_classes_to_show, :max_classes_to_show]
    plt.figure(figsize=(8,8)); plt.imshow(C, interpolation='nearest'); plt.title("Confusion Matrix"); plt.colorbar()
    tick_marks = np.arange(len(labels_show))
    plt.xticks(tick_marks, labels_show, rotation=90, fontsize=6); plt.yticks(tick_marks, labels_show, fontsize=6)
    plt.tight_layout(); plt.ylabel('True'); plt.xlabel('Predicted'); plt.savefig(out_png, dpi=160, bbox_inches="tight"); plt.close()

@torch.no_grad()
def retrieval_eval(train_embs, train_labels, test_embs, test_labels,
                   topk_list, ignore_unknown=True, return_percent=True, report=False):
    """
    Centroid retrieval (test -> train centroids) with robust label mapping.

    - If a test id isn't in train:
        - ignore_unknown=True: skip it (doesn't affect the denominator)
        - ignore_unknown=False: count as miss
    - report=True returns a dict with coverage/missing ids.
    """
    import numpy as np

    trX = np.asarray(train_embs,  dtype=np.float32)
    trY = np.asarray(train_labels, dtype=int)
    teX = np.asarray(test_embs,   dtype=np.float32)
    teY = np.asarray(test_labels, dtype=int)

    classes = np.unique(trY)                  # train id space
    class_to_idx = {int(c): i for i, c in enumerate(classes)}
    C = len(classes)

    # map test labels to centroid indices (or -1 if not present in train)
    teY_idx = np.array([class_to_idx.get(int(y), -1) for y in teY], dtype=int)
    known_mask = teY_idx >= 0

    # optional reporting
    missing_ids = np.unique(teY[~known_mask]).tolist()
    coverage = float(known_mask.mean()) if teY.size else 0.0

    # build centroids (rows ordered by 'classes')
    centroids = np.stack([trX[trY == c].mean(axis=0) for c in classes], axis=0)
    centroids /= (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-8)

    # normalize test
    X = teX / (np.linalg.norm(teX, axis=1, keepdims=True) + 1e-8)

    # cosine sims [N_test, C]
    sims = X @ centroids.T

    # top-k indices (in centroid index space)
    ranks = np.argsort(-sims, axis=1)

    out = {}
    for k in topk_list:
        k_eff = max(1, min(int(k), C))
        topk = ranks[:, :k_eff]                           # [N_test, k]
        hits = (topk == teY_idx[:, None]).any(axis=1)     # [N_test] bool

        if ignore_unknown:
            denom = max(1, known_mask.sum())
            acc = float(hits[known_mask].mean()) if denom > 0 else 0.0
        else:
            acc = float(hits.mean())

        out[k] = f"{100.0*acc:.2f}%" if return_percent else acc

    if report:
        return out, {"coverage": coverage, "n_test": int(teY.size),
                     "n_known": int(known_mask.sum()),
                     "n_unknown": int((~known_mask).sum()),
                     "unknown_ids": missing_ids}
    return out


# ---------- Verification (ROC/AUC, EER, TAR@FAR) ----------
def _sample_genuine_pairs(labels, max_pairs, rng):
    idx_by_lab = {}
    for i, y in enumerate(labels):
        idx_by_lab.setdefault(int(y), []).append(i)
    pairs = []
    labs = list(idx_by_lab.keys())
    if len(labs) == 0: return pairs
    while len(pairs) < max_pairs:
        lab = rng.choice(labs)
        idxs = idx_by_lab[lab]
        if len(idxs) < 2: continue
        i, j = rng.choice(idxs, size=2, replace=False)
        pairs.append((int(i), int(j)))
        if len(pairs) >= max_pairs: break
    return pairs

def _sample_impostor_pairs(labels, max_pairs, rng):
    n = len(labels); pairs = []; labels = np.asarray(labels)
    if n <= 1: return pairs
    while len(pairs) < max_pairs:
        i = int(rng.randint(0, n)); j = int(rng.randint(0, n))
        if i == j: continue
        if labels[i] != labels[j]: pairs.append((i, j))
    return pairs

def _cosine_scores(embs, pairs):
    a = np.array([embs[i] for i, _ in pairs], dtype=np.float32)
    b = np.array([embs[j] for _, j in pairs], dtype=np.float32)
    return np.sum(a * b, axis=1)

def _eer_from_roc(fpr, tpr, thr):
    fnr = 1 - tpr
    idx = np.argmin(np.abs(fpr - fnr))
    eer = 0.5 * (fpr[idx] + fnr[idx]); thr_eer = thr[idx]
    return float(eer), float(thr_eer)

def _tar_at_far(fpr, tpr, thr, far_target):
    mask = fpr <= far_target
    if not np.any(mask):
        best_idx = np.argmax(fpr)
        return float(tpr[best_idx]), float(thr[best_idx])
    idx = np.argmax(tpr[mask])
    return float(tpr[mask][idx]), float(thr[mask][idx])

@torch.no_grad()
def compute_verification_and_sim(
    out_dir: str,
    train_embs, train_lbls,
    test_embs,  test_lbls,
    cfg: dict
):
    """
    Fast, vectorized verification + retrieval.
    - Accepts torch tensors OR numpy arrays; auto-moves to CUDA if available.
    - Cosine similarity via normalized matmul (no Python loops).
    - Retrieval via torch.topk (no full sort).
    - Optional: save similarity matrix (.pt), sample CSV of pairs (slow -> off by default).
    - Proper TAR@FAR with linear interpolation (not nearest bin).
    """
    import os, json
    import numpy as np
    import torch
    import torch.nn.functional as F
    os.makedirs(out_dir, exist_ok=True)
    ver_dir = Path(out_dir) / "verification"
    ver_dir.mkdir(parents=True, exist_ok=True)
    # ---------------- device & coercion ----------------
    def _infer_device(*xs):
        for x in xs:
            if isinstance(x, torch.Tensor):
                return x.device
        use_gpu = bool(cfg.get("use_gpu", True)) and torch.cuda.is_available()
        return torch.device("cuda" if use_gpu else "cpu")

    device = _infer_device(train_embs, test_embs)

    def _to_tensor(x, dtype=torch.float32):
        if isinstance(x, torch.Tensor):
            return x.to(device=device, dtype=dtype, non_blocking=True)
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x).to(device=device, dtype=dtype, non_blocking=True)
        if isinstance(x, (list, tuple)):
            # assume list of arrays/tensors -> cat on dim 0
            xs = [_to_tensor(y, dtype=dtype) for y in x]
            return torch.cat(xs, dim=0)
        raise TypeError(f"Unsupported type: {type(x)}")

    def _to_labels(x):
        if isinstance(x, torch.Tensor):
            return x.to(device=device, dtype=torch.long, non_blocking=True)
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x).to(device=device, dtype=torch.long, non_blocking=True)
        if isinstance(x, (list, tuple)):
            return torch.tensor(x, device=device, dtype=torch.long)
        raise TypeError(f"Unsupported labels type: {type(x)}")

    train_embs = _to_tensor(train_embs, dtype=torch.float32)
    test_embs  = _to_tensor(test_embs,  dtype=torch.float32)
    train_lbls = _to_labels(train_lbls)
    test_lbls  = _to_labels(test_lbls)

    # sanity: [N, D]
    if train_embs.dim() == 1: train_embs = train_embs.unsqueeze(0)
    if test_embs.dim()  == 1: test_embs  = test_embs.unsqueeze(0)

    save_csv  = bool(cfg.get("save_verif_csv", False))
    save_mat  = bool(cfg.get("save_similarity_matrix", False))
    do_retr   = bool(cfg.get("do_retrieval", True))
    k_list    = cfg.get("retrieval_topk", [1, 5, 10])

    # ---------------- cosine normalization ----------------
    def _l2n(x):  # x: [N, D]
        return F.normalize(x, dim=1, eps=1e-8)

    tr = _l2n(train_embs)
    te = _l2n(test_embs)

    # ---------------- sims (test x train) ----------------
    # sims_te_tr: [N_test, N_train]
    sims_te_tr = te @ tr.T

    if save_mat:
        torch.save(
            {
                "sims_te_tr": sims_te_tr.float().cpu(),
                "train_lbls": train_lbls.cpu(),
                "test_lbls":  test_lbls.cpu(),
            },
            os.path.join(out_dir, "similarity_matrices.pt"),
        )

    # ---------------- verification ----------------
    # label mask by broadcasting
    te_ids = test_lbls.view(-1, 1)          # [N_test, 1]
    tr_ids = train_lbls.view(1, -1)         # [1, N_train]
    y_true = (te_ids == tr_ids)             # bool
    scores = sims_te_tr.reshape(-1)         # cosine scores
    labels = y_true.reshape(-1).to(torch.uint8)

    if save_csv:
        try:
            import pandas as pd
            n = min(int(cfg.get("max_pairs_csv", 100000)), scores.numel())
            idx = torch.randperm(scores.numel(), device=scores.device)[:n]
            df = pd.DataFrame({
                "score": scores[idx].float().cpu().numpy(),
                "label": labels[idx].int().cpu().numpy(),
            })
            df.to_csv(ver_dir / "verification_pairs_sample.csv", index=False)
        except Exception:
            pass  # CSV is optional; never let it block

    # AUC + TAR@FAR with linear interpolation
    metrics_verif = {"AUC": 0.0, "TAR@FAR=1e-2": 0.0, "TAR@FAR=1e-3": 0.0}
    try:
        from sklearn.metrics import roc_curve, auc
        s_np = scores.float().cpu().numpy()
        y_np = labels.cpu().numpy().astype(np.uint8)
        if s_np.size > 0 and y_np.max() != y_np.min():  # need both classes
            fpr, tpr, _ = roc_curve(y_np, s_np)
            roc_auc = float(auc(fpr, tpr))

            def _interp_tar(target_far: float) -> float:
                # linear interpolation on ROC curve
                if target_far <= fpr[0]:
                    return float(tpr[0])
                if target_far >= fpr[-1]:
                    return float(tpr[-1])
                i = np.searchsorted(fpr, target_far, side="left")
                x0, x1 = fpr[i-1], fpr[i]
                y0, y1 = tpr[i-1], tpr[i]
                if x1 == x0:
                    return float(y1)
                w = (target_far - x0) / (x1 - x0)
                return float(y0 * (1 - w) + y1 * w)

            metrics_verif = {
                "AUC": roc_auc,
                "TAR@FAR=1e-2": _interp_tar(1e-2),
                "TAR@FAR=1e-3": _interp_tar(1e-3),
            }
    except Exception:
        pass  # keep zeros if sklearn isn't available or any issue

    # ---------------- write summary ----------------
    summary = {"verification": metrics_verif}
    with open(ver_dir / "verification_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("[VERIFICATION]", summary)



# ---------- Gallery/Probe protocol on TEST ----------
def _build_gallery_probe_indices(labels: np.ndarray, g_per: int, selection: str, seed: int):
    """
    Build gallery/probe indices from a single split.
    - For each class with n>=2, takes g=min(g_per, n-1) images for gallery and the rest as probes.
    - selection in {"first","last","random"}.
    Returns: g_idx (list[int]), p_idx (list[int]).
    """
    labels = np.asarray(labels, dtype=int)
    rng = np.random.default_rng(seed)
    g_idx, p_idx = [], []

    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        n = idx.size
        if n < 2:
            continue  # cannot form gallery+probe from this class

        g_take = max(1, min(int(g_per), n - 1))
        if selection == "first":
            g = idx[:g_take]
            p = idx[g_take:]
        elif selection == "last":
            g = idx[-g_take:]
            p = idx[:-g_take]
        else:  # "random" (default/fallback)
            perm = rng.permutation(idx)
            g = perm[:g_take]
            p = perm[g_take:]

        # guard: ensure we still have at least 1 probe
        if p.size == 0:
            # move one from gallery to probe if needed
            p = g[-1:]; g = g[:-1]
        if g.size == 0 or p.size == 0:
            continue

        g_idx.extend(g.tolist())
        p_idx.extend(p.tolist())

    return g_idx, p_idx

import json, os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def _build_gallery_probe_indices(labels: np.ndarray, g_per: int, selection: str, seed: int):
    """
    Build gallery/probe indices from a single split.
    - For each class with n>=2, takes g=min(g_per, n-1) images for gallery and the rest as probes.
    - selection in {"first","last","random"}.
    Returns: g_idx (list[int]), p_idx (list[int]).
    """
    labels = np.asarray(labels, dtype=int)
    rng = np.random.default_rng(seed)
    g_idx, p_idx = [], []

    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        n = idx.size
        if n < 2:
            continue  # cannot form gallery+probe from this class

        g_take = max(1, min(int(g_per), n - 1))
        if selection == "first":
            g = idx[:g_take]
            p = idx[g_take:]
        elif selection == "last":
            g = idx[-g_take:]
            p = idx[:-g_take]
        else:  # "random" (default/fallback)
            perm = rng.permutation(idx)
            g = perm[:g_take]
            p = perm[g_take:]

        # guard: ensure we still have at least 1 probe
        if p.size == 0:
            # move one from gallery to probe if needed
            p = g[-1:]; g = g[:-1]
        if g.size == 0 or p.size == 0:
            continue

        g_idx.extend(g.tolist())
        p_idx.extend(p.tolist())

    return g_idx, p_idx

def compute_gallery_probe_identification(out_dir,
                                         test_embs, test_labels, test_items,
                                         cfg, label_to_person):
    """
    Robust gallery/probe identification.
    - If test split has too few images/class, auto-shrinks g_per to n-1.
    - If STILL no pairs, optionally falls back to train-as-gallery when provided via cfg["gallery_probe"].
      Expect keys: "train_embs", "train_labels", "train_items".
    - Saves lists, similarity matrix, heatmap, rank-k, and confusion matrix over seen labels.
    """
    gp_cfg = cfg.get("gallery_probe", {})
    if not gp_cfg.get("enabled", False):
        return

    gp_dir = Path(out_dir) / "gallery_probe"
    gp_dir.mkdir(parents=True, exist_ok=True)

    # ---------------- load + normalize ----------------
    emb_t = np.asarray(test_embs, dtype=np.float32)
    emb_t /= (np.linalg.norm(emb_t, axis=1, keepdims=True) + 1e-8)
    lab_t = np.asarray(test_labels, dtype=int)

    g_per = int(gp_cfg.get("gallery_per_class", 1))
    selection = str(gp_cfg.get("selection", "first")).lower()
    if selection not in {"first", "last", "random"}:
        selection = "random"
    seed = int(gp_cfg.get("seed", 42))
    topk_list = list(map(int, cfg.get("topk", [1, 5, 10])))

    # ---------------- try: gallery/probe within TEST ----------------
    g_idx, p_idx = _build_gallery_probe_indices(lab_t, g_per, selection, seed)

    # ---------------- fallback: use TRAIN as gallery (optional) ----------------
    # To enable, put train arrays into cfg["gallery_probe"]:
    # gp_cfg["use_train_as_gallery"] = True
    # gp_cfg["train_embs"], gp_cfg["train_labels"], gp_cfg["train_items"]
    used_train_gallery = False
    if (len(g_idx) == 0 or len(p_idx) == 0) and gp_cfg.get("use_train_as_gallery", False):
        tr_embs = gp_cfg.get("train_embs", None)
        tr_labs = gp_cfg.get("train_labels", None)
        tr_items = gp_cfg.get("train_items", None)
        if tr_embs is not None and tr_labs is not None and tr_items is not None:
            emb_g = np.asarray(tr_embs, dtype=np.float32)
            emb_g /= (np.linalg.norm(emb_g, axis=1, keepdims=True) + 1e-8)
            lab_g = np.asarray(tr_labs, dtype=int)
            # probes come from test
            emb_p = emb_t
            lab_p = lab_t
            g_idx = np.arange(emb_g.shape[0])
            p_idx = np.arange(emb_p.shape[0])
            used_train_gallery = True
        else:
            print("[GALLERY/PROBE] Train-as-gallery requested but train arrays not provided in cfg['gallery_probe'].")
            # will fall through to final guard

    # ---------------- final guard ----------------
    if (len(g_idx) == 0 or len(p_idx) == 0) and not used_train_gallery:
        print("[GALLERY/PROBE] Not enough samples to build protocol (need ≥2 images per identity in test). Skipping.")
        return

    # ---------------- assemble arrays ----------------
    if used_train_gallery:
        # already prepared above
        pass
    else:
        emb_g = emb_t[g_idx]; lab_g = lab_t[g_idx]
        emb_p = emb_t[p_idx]; lab_p = lab_t[p_idx]

    # ---------------- save lists (robust to path types) ----------------
    def _as_posix(p):
        try:
            return p.as_posix()
        except AttributeError:
            return str(p)

    if used_train_gallery:
        # train items come from cfg
        tr_items = gp_cfg["train_items"]
        g_rows = [{"idx": int(i), "label": int(l),
                   "person_id": label_to_person.get(int(l), int(l)),
                   "path": _as_posix(tr_items[int(i)][0])}
                  for i, l in zip(g_idx, lab_g)]
        p_rows = [{"idx": int(i), "label": int(l),
                   "person_id": label_to_person.get(int(l), int(l)),
                   "path": _as_posix(test_items[int(i)][0])}
                  for i, l in zip(p_idx, lab_p)]
    else:
        g_rows = [{"idx": int(i), "label": int(l),
                   "person_id": label_to_person.get(int(l), int(l)),
                   "path": _as_posix(test_items[int(i)][0])}
                  for i, l in zip(g_idx, lab_g)]
        p_rows = [{"idx": int(i), "label": int(l),
                   "person_id": label_to_person.get(int(l), int(l)),
                   "path": _as_posix(test_items[int(i)][0])}
                  for i, l in zip(p_idx, lab_p)]

    pd.DataFrame(g_rows).to_csv(gp_dir / "gallery_list.csv", index=False)
    pd.DataFrame(p_rows).to_csv(gp_dir / "probe_list.csv", index=False)

    # ---------------- probe → gallery similarity (cosine) ----------------
    S = np.matmul(emb_p, emb_g.T)  # [N_probe, N_gallery]
    np.save(gp_dir / "similarity_probe_gallery.npy", S)
    plt.figure(figsize=(7, 5))
    plt.imshow(S, aspect='auto', interpolation='nearest')
    plt.title("Cosine Similarity (probe→gallery)")
    plt.colorbar(); plt.tight_layout()
    plt.savefig(gp_dir / "similarity_probe_gallery_heatmap.png", dpi=160)
    plt.close()

    # ---------------- rank-k identification ----------------
    ranks = np.argsort(-S, axis=1)  # descending
    topk_metrics = {}
    for k in topk_list:
        k_eff = max(1, min(int(k), ranks.shape[1]))
        # hit if any of top-k predictions match probe label
        hit = np.any(lab_g[ranks[:, :k_eff]] == lab_p[:, None], axis=1)
        topk_metrics[f"rank@{k}"] = float(np.mean(hit))

    # ---------------- confusion matrix over seen labels ----------------
    # restrict to labels that appear in probes or predicted top-1
    top1 = ranks[:, 0]
    y_true = lab_p
    y_pred = lab_g[top1]
    seen = np.unique(np.concatenate([y_true, y_pred], axis=0))
    class_names = [str(label_to_person.get(int(i), int(i))) for i in seen.tolist()]

    # save confusion matrix png (safe even if labels are a subset)
    try:
        from sklearn.metrics import confusion_matrix
        import seaborn as sns
        cm = confusion_matrix(y_true, y_pred, labels=seen)
        plt.figure(figsize=(max(6, 0.25 * len(seen)), max(5, 0.25 * len(seen))))
        sns.heatmap(cm, annot=False, fmt='d', cmap="Blues",
                    xticklabels=class_names, yticklabels=class_names)
        plt.xlabel("Predicted"); plt.ylabel("True"); plt.tight_layout()
        plt.savefig(gp_dir / "confusion_matrix.png", dpi=160)
        plt.close()
    except Exception:
        # fall back to your existing helper if you prefer; this block is optional
        pass

    # ---------------- write summary ----------------
    with open(gp_dir / "identification_topk.json", "w") as f:
        json.dump({
            "used_train_as_gallery": bool(used_train_gallery),
            "gallery_per_class_requested": int(g_per),
            "selection": selection,
            "n_gallery": int(len(g_idx)) if not used_train_gallery else int(emb_g.shape[0]),
            "n_probes": int(len(p_idx)) if not used_train_gallery else int(emb_p.shape[0]),
            **topk_metrics
        }, f, indent=2)

    print("[GALLERY/PROBE]", {k: f"{v*100:.2f}%" for k, v in topk_metrics.items()})

# ---------------------
# Visualization exports (tight ablations)
# ---------------------
@torch.no_grad()
def _denorm(x, mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)):
    m = torch.tensor(mean, device=x.device).view(1,3,1,1)
    s = torch.tensor(std,  device=x.device).view(1,3,1,1)
    return torch.clamp(x * s + m, 0, 1)

def save_overlay(img_rgb, heat, out_path, alpha=0.5, cmap=cv2.COLORMAP_JET):
    """
    img_rgb: (H,W,3) float [0,1]
    heat: (H,W) float [0,1]
    """
    H, W = heat.shape
    heat8 = np.uint8(np.clip(heat, 0, 1) * 255)
    color = cv2.applyColorMap(heat8, cmap)[:, :, ::-1] / 255.0
    over = np.clip((1-alpha)*img_rgb + alpha*color, 0, 1)
    cv2.imwrite(str(out_path), np.uint8(over*255))

def compute_curvature(gray):
    # curvature of level sets: k = div(grad/|grad|)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx*gx + gy*gy) + 1e-8
    nx, ny = gx/mag, gy/mag
    nxx = cv2.Sobel(nx, cv2.CV_32F, 1, 0, ksize=3)
    nyy = cv2.Sobel(ny, cv2.CV_32F, 0, 1, ksize=3)
    k = nxx + nyy
    k = (k - k.min()) / (k.max() - k.min() + 1e-8)
    return k

def persistence_curve(edge_mag, thresholds=np.linspace(0.1,0.9,9)):
    """
    Very lightweight proxy: count connected components (Betti-0) across thresholds.
    """
    betti0 = []
    for t in thresholds:
        _, bw = cv2.threshold((edge_mag*255).astype(np.uint8), int(t*255), 255, cv2.THRESH_BINARY)
        num_labels, _ = cv2.connectedComponents(bw)
        betti0.append(num_labels-1)  # exclude background
    return thresholds, np.array(betti0, dtype=np.int32)

@torch.no_grad()
def export_visuals(model, dl, device, out_dir: Path, cfg: dict):
    viz_cfg = cfg.get("viz", {})
    if not viz_cfg.get("enabled", True): return
    max_batches = int(viz_cfg.get("max_batches", 2))
    vdir = out_dir / viz_cfg.get("out_dir", "viz")
    sub = {
        "contour": vdir / "contour",
        "grad": vdir / "grad",
        "curvature": vdir / "curvature",
        "persistence": vdir / "persistence",
        "attention": vdir / "attention"
    }
    for d in sub.values(): d.mkdir(parents=True, exist_ok=True)

    model.eval()
    bcount = 0
    for imgs, labels in dl:
        imgs = imgs.to(device)
        emb, logits, aux, _ = model.forward_with_aux(imgs, labels=None) if hasattr(model, "forward_with_aux") else (None, None, {}, {})
        B = imgs.size(0)
        den = _denorm(imgs)

        # contour overlays + gradient fields
        cont = aux.get("contour", None)
        if cont is not None:
            # gradient fields
            gx = cont[:,0:1]; gy = cont[:,1:2]; lap = cont[:,2:3]
            mag = torch.clamp((gx.abs() + gy.abs())/2.0, 0, 1)
            for i in range(B):
                rgb = (den[i].permute(1,2,0).cpu().numpy())
                m = mag[i,0].cpu().numpy()
                save_overlay(rgb, m, sub["contour"]/f"overlay_{bcount}_{i}.png", alpha=0.45)
                cv2.imwrite(str(sub["grad"]/f"gx_{bcount}_{i}.png"), np.uint8(np.clip(gx[i,0].cpu().numpy(),0,1)*255))
                cv2.imwrite(str(sub["grad"]/f"gy_{bcount}_{i}.png"), np.uint8(np.clip(gy[i,0].cpu().numpy(),0,1)*255))
                cv2.imwrite(str(sub["grad"]/f"mag_{bcount}_{i}.png"), np.uint8(np.clip(m,0,1)*255))
                cv2.imwrite(str(sub["grad"]/f"lap_{bcount}_{i}.png"), np.uint8(np.clip(lap[i,0].cpu().numpy(),0,1)*255))

                # curvature
                gray = cv2.cvtColor(np.uint8(np.clip(rgb*255,0,255)), cv2.COLOR_RGB2GRAY).astype(np.float32)/255.0
                curv = compute_curvature(gray)
                cv2.imwrite(str(sub["curvature"]/f"curv_{bcount}_{i}.png"), np.uint8(curv*255))

                # persistence curve (Betti-0) from mag
                th, b0 = persistence_curve(m)
                plt.figure(figsize=(3.0,2.2)); plt.plot(th, b0, marker='o'); plt.grid(True, alpha=0.3)
                plt.xlabel("threshold"); plt.ylabel("Betti-0"); plt.tight_layout()
                plt.savefig(sub["persistence"]/f"betti0_{bcount}_{i}.png", dpi=140); plt.close()

        # attention: CBAM spatial map (if any)
        cbam_sa = aux.get("cbam_sa", None)
        if cbam_sa is not None:
            sa = cbam_sa
            for i in range(min(B, sa.size(0))):
                A = F.interpolate(sa[i:i+1], size=den.shape[-2:], mode="bilinear", align_corners=False)[0,0].cpu().numpy()
                rgb = (den[i].permute(1,2,0).cpu().numpy())
                save_overlay(rgb, (A - A.min())/(A.max()-A.min()+1e-8), sub["attention"]/f"cbam_sa_{bcount}_{i}.png", alpha=0.45)

        # attention: ECA channel weights (bar chart)
        eca_w = aux.get("eca_w", None)
        if eca_w is not None:
            w = eca_w.mean(dim=0).view(-1).cpu().numpy()  # mean over batch -> (C,)
            plt.figure(figsize=(6,2)); plt.plot(w); plt.title("ECA channel weights (mean batch)")
            plt.tight_layout(); plt.savefig(sub["attention"]/f"eca_weights_{bcount}.png", dpi=150); plt.close()

        bcount += 1
        if bcount >= max_batches: break

# ---------------------
# Train / Eval loops (torch.amp API) with EDH/ONSA integration
# ---------------------
def arcface_ms_schedule(epoch: int, total_epochs: int,
                        m_warm_start: float = 0.10, m_peak: float = 0.35,
                        warmup_epochs: int = 20, decay_start: int = 80, m_end: float = 0.30,
                        s_start: float = 64.0, s_target: float = 76.0, s_warmup: int = 10) -> tuple[float, float]:
    """
    Margin m:  linear ramp 0.10 → 0.35 over first ~20 epochs, hold, then decay to 0.30 after epoch ~80.
    Scale  s:  gentle ramp 64 → 76 over first ~10 epochs (then fixed). Try 72–80 by changing s_target.
    """
    # --- margin ---
    if epoch <= warmup_epochs:
        t = (epoch - 1) / max(1, warmup_epochs - 1)
        m = m_warm_start + t * (m_peak - m_warm_start)
    elif epoch <= decay_start:
        m = m_peak
    else:
        t = (epoch - decay_start) / max(1, total_epochs - decay_start)
        t = min(max(t, 0.0), 1.0)
        m = m_peak + t * (m_end - m_peak)

    # --- scale ---
    if epoch <= s_warmup:
        t = (epoch - 1) / max(1, s_warmup - 1)
        s = s_start + t * (s_target - s_start)
    else:
        s = s_target
    return float(m), float(s)


def train_one_epoch(model, dl, optimizer, scaler, device, epoch, log_file, use_amp_device, cfg):
    model.train()
    ce = nn.CrossEntropyLoss()
    total_loss, total_acc1, total_acc5, n_samples = 0.0, 0.0, 0.0, 0

    # --- ArcFace margin/scale schedule for this epoch ---
    m_cur, s_cur = arcface_ms_schedule(epoch, cfg["epochs"])

    try:
        model.head.s = float(s_cur)
        model.s = float(s_cur)  # optional: some utilities may read model.s
    except Exception:
        pass

    for imgs, labels in dl:
        imgs = imgs.to(device); labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)

        # ---- forward with margin for loss/backprop ----
        with torch.amp.autocast(use_amp_device, enabled=(use_amp_device == 'cuda')):
            emb, logits, aux, extra = model.forward_with_aux(imgs, labels, current_m=m_cur, current_s=s_cur)
            loss = ce(logits, labels)
            if "edh_l1" in extra:  loss = loss + extra["edh_l1"]
            if "edh_eik" in extra: loss = loss + extra["edh_eik"]
            if "onsa_pen" in extra: loss = loss + extra["onsa_pen"]
            if "onsa_adv" in extra: loss = loss + extra["onsa_adv"]

        if scaler is not None:
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
        else:
            loss.backward(); optimizer.step()

        # ---- accuracy for logging (NO-MARGIN) ----
        with torch.no_grad():
            Wn = F.normalize(model.head.weight, dim=1)  # (C, D)
            logits_nomargin = (emb @ Wn.t()) * float(s_cur)
            acc1, acc5 = accuracy_topk(logits_nomargin, labels, ks=(1,5))

        # ---- accounting ----
        bsz = labels.size(0)
        total_loss += loss.item() * bsz
        total_acc1 += acc1 * bsz / 100.0
        total_acc5 += acc5 * bsz / 100.0
        n_samples += bsz

    avg_loss = total_loss / max(1, n_samples)
    avg_acc1 = (total_acc1 / max(1, n_samples)) * 100.0
    avg_acc5 = (total_acc5 / max(1, n_samples)) * 100.0
    log_file.write(f"{epoch},train,{avg_loss:.4f},{avg_acc1:.2f},{avg_acc5:.2f}\n"); log_file.flush()
    return avg_loss, avg_acc1


@torch.no_grad()
def eval_logits_no_margin(model, imgs):
    emb, _ = model(imgs, labels=None)             # get normalized embeddings
    W = torch.nn.functional.normalize(model.head.weight, dim=1)
    s = float(getattr(model.head, "s", 1.0))
    return emb @ W.t() * s

@torch.no_grad()
def evaluate(model, dl, device, epoch, split, log_file, use_amp_device) -> Tuple[float, float, np.ndarray, np.ndarray]:
    model.eval()
    ce = nn.CrossEntropyLoss()
    total_loss, total_acc1, n_samples = 0.0, 0.0, 0
    all_labels, all_preds = [], []
    for imgs, labels in dl:
        imgs = imgs.to(device); labels = labels.to(device)
        # _, logits = model(imgs, labels)   # plain forward: classification only
        logits = eval_logits_no_margin(model, imgs)
        loss = ce(logits, labels)
        acc1, acc5 = accuracy_topk(logits, labels, ks=(1,5))
        bsz = labels.size(0)
        total_loss += loss.item() * bsz
        total_acc1 += acc1 * bsz / 100.0
        total_acc5 = locals().get('total_acc5', 0.0) + (acc5 * bsz / 100.0)
        locals().update({'total_acc5': total_acc5})
        n_samples += bsz
        all_labels.append(labels.cpu().numpy())
        all_preds.append(torch.argmax(logits, dim=1).cpu().numpy())
    avg_loss = total_loss / max(1, n_samples)
    avg_acc1 = (total_acc1 / max(1, n_samples)) * 100.0
    avg_acc5 = (locals().get('total_acc5', 0.0) / max(1, n_samples)) * 100.0
    all_labels = np.concatenate(all_labels) if all_labels else np.array([])
    all_preds  = np.concatenate(all_preds)  if all_preds  else np.array([])
    log_file.write(f"{epoch},{split},{avg_loss:.4f},{avg_acc1:.2f},{avg_acc5:.2f}\n"); log_file.flush()
    return avg_loss, avg_acc1, all_labels, all_preds

# ---------------------
# Main
# ---------------------
def scan_dataset(ds):
    print("[SCAN] Scanning train dataset sequentially…")
    for i in range(len(ds)):
        try:
            _ = ds[i]    # triggers _crop_and_cache + transforms
        except Exception as e:
            print(f"[SCAN][FAIL] idx={i} path={ds.samples[i][0]} :: {e}")
            raise
    print("[SCAN] OK — no per-item error")

def main():
    import os, platform, torch
    if platform.system() == "Windows":
        try:
            torch.multiprocessing.set_start_method("spawn", force=True)
        except RuntimeError:
            pass
        os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

    class _Tee(io.TextIOBase):
        def __init__(self, *streams): self._s = streams
        def write(self, b):
            for s in self._s: s.write(b)
            return len(b)
        def flush(self):
            for s in self._s:
                try: s.flush()
                except Exception: pass

    cfg = CONFIG.copy()
    set_seed(cfg["seed"])

    device = cfg["device"]
    # --- Device info print ---
    if torch.cuda.is_available() and device.startswith("cuda"):
        idx = 0 if device == "cuda" else int(device.split(":")[1])
        props = torch.cuda.get_device_properties(idx)
        print(f"[DEVICE] Using CUDA:{idx} → {props.name}, "
              f"{props.total_memory/1024**3:.1f} GB VRAM, "
              f"compute capability {props.major}.{props.minor}")
    else:
        print(f"[DEVICE] Using {device} (CPU)")
        
    out_dir = Path(cfg["work_dir"])
    ensure_dir(out_dir); ensure_dir(out_dir / "embeddings")

    _result_f = open(out_dir / "result.txt", "a", buffering=1, encoding="utf-8")
    _old_stdout, _old_stderr = sys.stdout, sys.stderr
    sys.stdout = _Tee(_old_stdout, _result_f)
    sys.stderr = _Tee(_old_stderr, _result_f)

    try: 
        with open(out_dir / "config.json", "w") as f: json.dump(cfg, f, indent=2)

        # Build splits
        train_items, val_items, test_items, label_map = split_dataset(cfg)
        with open(out_dir / "label_map.json", "w") as f: json.dump(label_map, f, indent=2)
        with open(out_dir / "splits.json", "w") as f:
            json.dump({
                "train": [(p.as_posix(), pid) for p, pid in train_items],
                "val":   [(p.as_posix(), pid) for p, pid in val_items],
                "test":  [(p.as_posix(), pid) for p, pid in test_items],
            }, f, indent=2)

        id2label = {v:k for k,v in label_map.items()}
        num_classes = len(label_map)
        print(f"[INFO] Persons: {num_classes} | Train: {len(train_items)} | Val: {len(val_items)} | Test: {len(test_items)}")

        # Ear detector
        detector = None
        if cfg["use_yolo_crop"] and cfg["ear_detector_weights"]:
            detector = EarDetector(
                            weights_path=CONFIG["ear_detector_weights"],
                            conf=CONFIG.get("yolo_conf", 0.25),
                            iou=CONFIG.get("yolo_iou", 0.50),
                            pad=CONFIG.get("yolo_pad", 0.02),
                            square=CONFIG.get("yolo_square", False),
                            device=CONFIG.get("yolo_device", None)
                        )
            if detector is not None and detector.enabled:
                print("[INFO] YOLO ear detector ENABLED.")
            else:
                print("[INFO] YOLO ear detector DISABLED (fallback to center-crop).")
        else:
            print("[INFO] YOLO ear detector not configured. Using center-crop fallback.")

        # Datasets / Loaders
        train_ds = EarVNDataset(train_items, label_map, cfg, stage="train", detector=detector)
        val_ds   = EarVNDataset(val_items, label_map, cfg, stage="val", detector=detector)
        test_ds  = EarVNDataset(test_items, label_map, cfg, stage="test", detector=detector)

        from torch.utils.data import WeightedRandomSampler
        counts = np.bincount([label_map[pid] for _, pid in train_items], minlength=len(label_map))
        w_class = 1.0 / np.maximum(counts, 1)
        w_sample = [w_class[label_map[pid]] for _, pid in train_items]
        sampler = WeightedRandomSampler(w_sample, num_samples=len(w_sample), replacement=True)

        train_dl = DataLoader(train_ds, batch_size=CONFIG["batch_size"], sampler=sampler,
                       num_workers=CONFIG["num_workers"], pin_memory=True)
        val_dl   = DataLoader(val_ds,   batch_size=cfg["batch_size"], shuffle=False,
                            num_workers=cfg["num_workers"], pin_memory=True)
        test_dl  = DataLoader(test_ds,  batch_size=cfg["batch_size"], shuffle=False,
                            num_workers=cfg["num_workers"], pin_memory=True)

        # Model / Optim / AMP
        device = cfg["device"]
        model = ArcFaceClassifier(num_classes=num_classes,
                                embedding_dim=cfg["embedding_dim"],
                                s=cfg["arcface_s"], m=cfg["arcface_m"],
                                use_attention=cfg["use_attention"],
                                attention_type=cfg["attention_type"],
                                backbone_type=cfg.get("backbone_type", "two_stream"),
                                eca_k=cfg.get("eca_k", 3),
                                edh_cfg=cfg.get("edh", {}),
                                onsa_cfg=cfg.get("onsa", {})).to(device)
        load_pretrained_if_exists(model, cfg)
        # Optional: freeze trunk for 2 epochs to stabilize ArcFace head
        if hasattr(model.backbone, "trunk"):
            for p in model.backbone.trunk.parameters():
                p.requires_grad = False
        unfreeze_epoch = 2

        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
        from torch.optim.lr_scheduler import CosineAnnealingLR
        scheduler = CosineAnnealingLR(optimizer, T_max=CONFIG["epochs"], eta_min=1e-6)
        warmup_epochs = 3
        use_cuda = device.startswith("cuda")
        use_amp_device = 'cuda' if use_cuda and cfg["fp16"] else 'cpu'
        scaler = torch.amp.GradScaler(use_amp_device, enabled=(use_amp_device == 'cuda'))

        # Logs
        log_path = out_dir / "train_log.csv"
        with open(log_path, "w") as f: f.write("epoch,split,loss,acc1,acc5\n")

        best_val_acc, best_path = -1.0, None
        train_losses, val_losses, train_accs, val_accs = [], [], [], []

        # Training loop
        for epoch in range(1, cfg["epochs"]+1):
            with open(log_path, "a") as lf:
                tr_loss, tr_acc = train_one_epoch(model, train_dl, optimizer, scaler, device, epoch, lf, use_amp_device, cfg)
                va_loss, va_acc, _, _ = evaluate(model, val_dl, device, epoch, "val", lf, use_amp_device)

            train_losses.append(tr_loss); train_accs.append(tr_acc)
            val_losses.append(va_loss);   val_accs.append(va_acc)
            print(f"[EPOCH {epoch:03d}/{cfg['epochs']}] train_loss={tr_loss:.4f} acc1={tr_acc:.2f} | val_loss={va_loss:.4f} acc1={va_acc:.2f}")

            # Save checkpoints
            if va_acc > best_val_acc:
                best_val_acc = va_acc
                best_path = out_dir / "best.ckpt"
                torch.save({"epoch": epoch, "state_dict": model.state_dict(), "val_acc1": va_acc}, best_path)
            if (epoch % cfg["save_every"]) == 0 or epoch == cfg["epochs"]:
                last_path = out_dir / "last.ckpt"
                torch.save({"epoch": epoch, "state_dict": model.state_dict(), "val_acc1": va_acc}, last_path)

            # warmup + cosine
            if epoch <= warmup_epochs:
                for g in optimizer.param_groups:
                    g["lr"] = CONFIG["lr"] * epoch / max(1, warmup_epochs)
            else:
                scheduler.step()

            # Unfreeze backbone
            if epoch == unfreeze_epoch and hasattr(model.backbone, "trunk"):
                for p in model.backbone.trunk.parameters():
                    p.requires_grad = True

        # Curves
        plot_curves(train_losses, val_losses, train_accs, val_accs, out_png=out_dir / "curves_loss_acc.png")

        # Test (best model if available)
        if best_path and best_path.exists():
            try:
                ckpt = torch.load(best_path, map_location=device, weights_only=True)
            except TypeError:
                ckpt = torch.load(best_path, map_location=device)
            model.load_state_dict(ckpt["state_dict"])
            print(f"[INFO] Loaded best.ckpt (val_acc1={ckpt.get('val_acc1', -1):.2f})")

        # --- Visual ablations / exports ---
        export_visuals(model, test_dl, device, out_dir, cfg)

        with open(log_path, "a") as lf:
            te_loss, te_acc, y_true, y_pred = evaluate(model, test_dl, device, 0, "test", lf, use_amp_device)
        print(f"[TEST] loss={te_loss:.4f} acc1={te_acc:.2f}")

        # Confusion matrix (classifier predictions)
        labels_names = [id2label[i] for i in range(num_classes)]
        save_confusion_matrix_png(y_true, y_pred, labels_names, out_png=out_dir / "confusion_matrix.png")

        # Classification report CSV
        report = classification_report(y_true, y_pred, labels=list(range(num_classes)),
                                    target_names=labels_names, zero_division=0, output_dict=True)
        pd.DataFrame(report).to_csv(out_dir / "classification_report.csv")

        # Misclassified table
        mis_idx = np.where(y_true != y_pred)[0]
        pd.DataFrame({"idx": mis_idx, "y_true": y_true[mis_idx], "y_pred": y_pred[mis_idx]}).to_csv(
            out_dir / "samples_misclassified.csv", index=False
        )

        # Embeddings & retrieval
        train_embs, train_lbls = extract_embeddings(model, train_dl, device)
        test_embs,  test_lbls  = extract_embeddings(model, test_dl,  device)
        np.save(out_dir / "embeddings" / "train_embeddings.npy", train_embs)
        np.save(out_dir / "embeddings" / "train_labels.npy",     train_lbls)
        np.save(out_dir / "embeddings" / "test_embeddings.npy",  test_embs)
        np.save(out_dir / "embeddings" / "test_labels.npy",      test_lbls)

        metrics, rpt = retrieval_eval(
            train_embs, train_lbls,
            test_embs, test_lbls,
            topk_list=[1,5,10],
            ignore_unknown=True,
            return_percent=True,
            report=True
        )
        print("[RETRIEVAL]", metrics)
        print(f"[INFO] coverage={rpt['coverage']*100:.1f}% "
            f"| known={rpt['n_known']} unknown={rpt['n_unknown']}")
        if rpt["n_unknown"] > 0:
            print("[INFO] test IDs not in train:", rpt["unknown_ids"][:20], "...")

        # --- NEW: Full verification artifacts (ROC, hist, AUC, EER, TAR@FAR+thr, sim matrices) ---
        try:
            # coerce to numpy and L2-normalize (safety)
            def _to_np(x):
                if torch.is_tensor(x): x = x.detach().cpu().numpy()
                return x
            train_embs_np = _to_np(train_embs).astype(np.float32)
            test_embs_np  = _to_np(test_embs ).astype(np.float32)
            train_lbls_np = _to_np(train_lbls).astype(np.int32)
            test_lbls_np  = _to_np(test_lbls ).astype(np.int32)
            # L2 norm
            def _l2n(a):
                n = np.linalg.norm(a, axis=1, keepdims=True) + 1e-12
                return a / n
            train_embs_np = _l2n(train_embs_np)
            test_embs_np  = _l2n(test_embs_np)

            gp_cfg = cfg.get("gallery_probe", {})
            use_train_as_gallery = bool(gp_cfg.get("use_train_as_gallery", False))

            eval_dir_full = Path(out_dir) / "verification"
            eval_dir_full.mkdir(parents=True, exist_ok=True)

            summary_full = save_full_verification_artifacts(
                test_feats=test_embs_np,
                test_labels=test_lbls_np,
                gallery_feats=train_embs_np if use_train_as_gallery else None,
                gallery_labels=train_lbls_np if use_train_as_gallery else None,
                out_dir=str(eval_dir_full),
                cfg=CONFIG
            )
            print(f"[INFO] Full verification artifacts saved to: {eval_dir_full}")
            print(f"[VERIFICATION] AUC={summary_full['AUC']:.4f} | EER={summary_full['EER']:.4f}")

        except Exception as e:
            print(f"[WARN] Full verification artifacts skipped: {e}")

        # --- Verification + similarity matrices (existing light version) ---
        if cfg.get("do_verification", True):
            compute_verification_and_sim(out_dir, train_embs, train_lbls, test_embs, test_lbls, cfg)

        # --- Gallery/Probe protocol on TEST ---
        if cfg["gallery_probe"].get("enabled", True):
            label_to_person = {i: labels_names[i] for i in range(num_classes)}
            cfg["gallery_probe"].update({
                "train_embs": train_embs.cpu().numpy() if torch.is_tensor(train_embs) else train_embs,
                "train_labels": train_lbls.cpu().numpy() if torch.is_tensor(train_lbls) else train_lbls,
                "train_items": train_items,     # list of (Path, label, …) like test_items
            })
            compute_gallery_probe_identification(out_dir, test_embs, test_lbls, test_items, cfg, label_to_person)

        # ---- Open-set & calibration (from v2) ----
        try:
            # train embeddings for centroids
            train_embs, train_labels = extract_embeddings(model, train_dl, device)
            calibrate_and_apply_open_set(
                model=model,
                train_embs=train_embs,
                train_labels=train_labels,
                val_dl=val_dl,
                test_dl=test_dl,
                out_dir=out_dir,
                cfg=CONFIG
            )
            print("[INFO] Open-set calibration & reports saved under:", (Path(out_dir) / "open_set"))
        except Exception as e:
            print(f"[WARN] Open-set calibration skipped due to error: {e}")
            
        print(f"[DONE] Results saved to: {out_dir}")
        pass
    finally:
        sys.stdout, sys.stderr = _old_stdout, _old_stderr
        try: _result_f.close()
        except Exception: pass


if __name__ == "__main__":
    main()