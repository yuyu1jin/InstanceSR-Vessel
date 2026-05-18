import argparse
import csv
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List

import cv2
import numpy as np
import torch
import torch_npu

from model.generator import Generator
from model.stage2_head import Stage2Head
from utils.paths import get_dir, get_file
from utils.resize_pad import Letterbox


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
CLASS_NAMES = ["jetski", "rubber", "sailboat", "yacht"]
LR_SIZE = 32
SR_SIZE = 128
SR_SCALE = 4


def iter_images(path: Path) -> Iterable[Path]:
    if path.is_file() and path.suffix.lower() in IMG_EXTS:
        yield path
        return

    for image_path in sorted(path.rglob("*")):
        if image_path.is_file() and image_path.suffix.lower() in IMG_EXTS:
            yield image_path


def read_rgb(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"Failed to read image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def to_tensor_01(img_rgb: np.ndarray) -> torch.Tensor:
    x = img_rgb.astype(np.float32) / 255.0
    x = np.transpose(x, (2, 0, 1))
    return torch.from_numpy(x)


def to_uint8_rgb(x01: torch.Tensor) -> np.ndarray:
    if x01.dim() == 4:
        x01 = x01[0]
    x01 = x01.detach().clamp(0, 1).cpu().numpy()
    x01 = np.transpose(x01, (1, 2, 0))
    return (x01 * 255.0 + 0.5).astype(np.uint8)


def save_rgb(path: Path, img_rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), bgr)


class InstanceSRRunner:
    def __init__(self, checkpoint: Path, device: str):
        self.device = torch.device(device if device == "npu" and torch.npu.is_available() else "cpu")
        self.letterbox = Letterbox(
            target_w=LR_SIZE,
            target_h=LR_SIZE,
            pad_mode="sea",
            center=True,
            allow_scale_up=True,
            max_downscale=6.0,
        )

        self.net_g = Generator().to(self.device).eval()
        self.net_h = Stage2Head(pretrained=True, num_classes=len(CLASS_NAMES)).to(self.device).eval()

        ckpt = torch.load(str(checkpoint), map_location=self.device)
        self.net_g.load_state_dict(ckpt["net_g"], strict=True)
        self.net_h.load_state_dict(ckpt["net_h"], strict=True)

    @torch.no_grad()
    def run_one(self, image_path: Path) -> Dict:
        rgb = read_rgb(image_path)
        lr_rgb, meta, over_compress = self.letterbox(rgb)
        lr = to_tensor_01(lr_rgb).unsqueeze(0).to(self.device)

        sr = self.net_g(lr).clamp(0, 1)
        cls_logits, box_refine = self.net_h(sr)

        prob = torch.softmax(cls_logits, dim=1)[0]
        pred_cls = int(torch.argmax(prob).item())
        pred_prob = float(prob[pred_cls].item())
        delta_w, delta_h = [float(x) for x in box_refine.squeeze(0).tolist()]

        pred_w_sr = float(np.clip(SR_SIZE * (1.0 - delta_w), 1.0, SR_SIZE))
        pred_h_sr = float(np.clip(SR_SIZE * (1.0 - delta_h), 1.0, SR_SIZE))
        pred_w_input = pred_w_sr / (SR_SCALE * max(meta.scale, 1e-6))
        pred_h_input = pred_h_sr / (SR_SCALE * max(meta.scale, 1e-6))

        return {
            "image": str(image_path),
            "pred_class_id": pred_cls,
            "pred_class_name": CLASS_NAMES[pred_cls],
            "pred_prob": pred_prob,
            "delta_w": delta_w,
            "delta_h": delta_h,
            "pred_w_sr": pred_w_sr,
            "pred_h_sr": pred_h_sr,
            "pred_w_input": float(np.clip(pred_w_input, 1.0, rgb.shape[1])),
            "pred_h_input": float(np.clip(pred_h_input, 1.0, rgb.shape[0])),
            "over_compress": bool(over_compress),
            "letterbox": meta.__dict__,
            "sr_rgb": to_uint8_rgb(sr),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Instance-level SR reconstruction inference.")
    parser.add_argument("--input", type=Path, default=get_dir("inference.input_dir"))
    parser.add_argument("--output", type=Path, default=get_dir("inference.output_dir", create=True))
    parser.add_argument("--checkpoint", type=Path, default=get_file("weights.stage2_weights"))
    parser.add_argument("--device", default="npu", choices=["npu", "cpu"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.output
    sr_dir = out_dir / "sr"
    json_dir = out_dir / "json"
    rows: List[Dict] = []

    runner = InstanceSRRunner(args.checkpoint, args.device)
    image_paths = list(iter_images(args.input))
    if not image_paths:
        raise RuntimeError(f"No instance images found in: {args.input}")

    for idx, image_path in enumerate(image_paths, 1):
        result = runner.run_one(image_path)
        rel = image_path.relative_to(args.input) if args.input.is_dir() else Path(image_path.name)
        stem = rel.with_suffix("")

        save_rgb(sr_dir / rel, result.pop("sr_rgb"))
        json_path = json_dir / stem.with_suffix(".json")
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        rows.append({k: v for k, v in result.items() if k != "letterbox"})
        print(f"[{idx}/{len(image_paths)}] {rel} -> {result['pred_class_name']} {result['pred_prob']:.4f}")

    csv_path = out_dir / "predictions.csv"
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Done. Results saved to: {out_dir}")


if __name__ == "__main__":
    main()
