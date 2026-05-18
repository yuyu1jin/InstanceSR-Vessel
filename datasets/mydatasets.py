from __future__ import annotations
import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch_npu
from torch.utils.data import Dataset


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _imread_rgb(path: Path) -> np.ndarray:
    img_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"cv2.imread failed: {path}")
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def _to_tensor(img_rgb: np.ndarray) -> torch.Tensor:
    x = img_rgb.astype(np.float32) / 255.0
    x = np.transpose(x, (2, 0, 1))
    return torch.from_numpy(x)


def _iter_images(root: Path, recursive: bool) -> List[Path]:
    if recursive:
        paths = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS]
    else:
        paths = [p for p in root.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS]
    paths.sort()
    return paths

def load_meta_dir(meta_dir: Path):
    meta = {}

    for csv_file in meta_dir.glob("*.csv"):

        name = csv_file.stem.lower()

        if "jetski" in name:
            cls = "jetski"
        elif "rubber" in name:
            cls = "rubber"
        elif "sailboat" in name:
            cls = "sailboat"
        elif "yacht" in name:
            cls = "yacht"
        else:
            continue

        meta[cls] = {}

        with open(csv_file, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)

            for row in reader:
                img = row["image"]
                w = float(row["new_w"]) 
                h = float(row["new_h"]) 

                meta[cls][img] = (w, h)

    return meta


class MyDataset(Dataset):

    def __init__(
        self,
        lq_root: str,
        gt_root: str,
        meta_root: str,
        hr_scale: float = 4.0,
        recursive: bool = False,
    ):

        self.lq_root = Path(lq_root)
        self.gt_root = Path(gt_root)
        self.meta_dir = Path(meta_root)
        self.hr_scale = hr_scale
        self.recursive = recursive

        self.class_to_label = {
            "jetski": 0,
            "rubber": 1,
            "sailboat": 2,
            "yacht": 3,
        }

        if not self.meta_dir.exists():
            raise FileNotFoundError(meta_dir)

        self.meta = load_meta_dir(self.meta_dir)

        self.samples = self.build_samples()

        if len(self.samples) == 0:
            raise RuntimeError("dataset empty")

    def build_samples(self):

        samples = []

        for cls in self.class_to_label:

            lq_dir = self.lq_root / cls
            gt_dir = self.gt_root / cls

            if not lq_dir.exists():
                continue

            label = self.class_to_label[cls]

            for img_path in _iter_images(lq_dir, self.recursive):

                name = img_path.name
                gt_path = gt_dir / name

                if not gt_path.exists():
                    continue

                if name not in self.meta.get(cls, {}):
                    continue

                hr_w, hr_h = self.meta[cls][name]

                samples.append((img_path, gt_path, label, hr_w, hr_h))

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        lq_path, gt_path, label, hr_w, hr_h = self.samples[idx]

        lq = _to_tensor(_imread_rgb(lq_path))
        gt = _to_tensor(_imread_rgb(gt_path))

        return {
            "lq": lq,
            "gt": gt,
            "label": torch.tensor(label, dtype=torch.long),
            "w": torch.tensor(hr_w, dtype=torch.float32),
            "h": torch.tensor(hr_h, dtype=torch.float32),
        }