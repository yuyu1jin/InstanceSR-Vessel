import os
import csv
import json
import math
from dataclasses import dataclass
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch
import torch_npu

from utils.paths import get_dir, get_file

TESTSET_ROOT = str(get_dir("experiments.partc_testset"))
CKPT_PATH = get_file("weights.stage2_weights")
OUT_DIR = str(get_dir("experiments.partc_output", create=True))

DEVICE = "npu"
NUM_CLASSES = 4
LR_SIZE = 32
SR_SIZE = 128   # 32 * 4
SAVE_VIS = True
MAX_VIS = 300

# category mapping
CLASS_TO_ID = {
    "jetski": 0,
    "rubber": 1,
    "sailboat": 2,
    "yacht": 3,
}
ID_TO_CLASS = {v: k for k, v in CLASS_TO_ID.items()}

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")

from model.generator import Generator
from model.stage2_head import Stage2Head

# utility function
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def to_tensor_01(img_rgb_u8: np.ndarray) -> torch.Tensor:
    x = img_rgb_u8.astype(np.float32) / 255.0
    x = np.transpose(x, (2, 0, 1))
    return torch.from_numpy(x)

def to_uint8_rgb(x01: torch.Tensor) -> np.ndarray:
    if x01.dim() == 4:
        x01 = x01[0]
    x01 = x01.detach().clamp(0, 1).cpu().numpy()
    x01 = np.transpose(x01, (1, 2, 0))
    return (x01 * 255.0 + 0.5).astype(np.uint8)

def clamp_float(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def center_box_iou(pred_w: float, pred_h: float, gt_w: float, gt_h: float) -> float:
    pred_w = max(0.0, pred_w)
    pred_h = max(0.0, pred_h)
    gt_w = max(0.0, gt_w)
    gt_h = max(0.0, gt_h)
    inter = min(pred_w, gt_w) * min(pred_h, gt_h)
    union = pred_w * pred_h + gt_w * gt_h - inter
    if union <= 0:
        return 0.0
    return float(inter / union)

def draw_center_box(img_rgb: np.ndarray, bw: float, bh: float, color=(255, 0, 0), thickness=2):
    out = img_rgb.copy()
    h, w = out.shape[:2]
    cx, cy = w / 2.0, h / 2.0

    bw = float(np.clip(bw, 1.0, w))
    bh = float(np.clip(bh, 1.0, h))

    x1 = int(round(cx - bw / 2.0))
    y1 = int(round(cy - bh / 2.0))
    x2 = int(round(cx + bw / 2.0))
    y2 = int(round(cy + bh / 2.0))

    x1 = max(0, min(w - 1, x1))
    y1 = max(0, min(h - 1, y1))
    x2 = max(0, min(w - 1, x2))
    y2 = max(0, min(h - 1, y2))

    bgr = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
    cv2.rectangle(bgr, (x1, y1), (x2, y2), (int(color[2]), int(color[1]), int(color[0])), thickness)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

def put_text(img_rgb: np.ndarray, lines, org=(5, 18), line_h=18):
    out = img_rgb.copy()
    bgr = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
    x0, y0 = org
    for i, s in enumerate(lines):
        y = y0 + i * line_h
        cv2.putText(bgr, s, (x0, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 0), 1, cv2.LINE_AA)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

def save_rgb(path: str, img_rgb: np.ndarray):
    ensure_dir(os.path.dirname(path))
    bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, bgr)

def compute_prf1(y_true: List[int], y_pred: List[int], num_classes: int):
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[t, p] += 1
    per_class = []
    for c in range(num_classes):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        per_class.append({
            "class_id": c,
            "class_name": ID_TO_CLASS.get(c, str(c)),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "support": int(cm[c, :].sum()),
        })
    return {
        "accuracy": float(np.trace(cm) / max(1, cm.sum())),
        "macro_precision": float(np.mean([x["precision"] for x in per_class])),
        "macro_recall": float(np.mean([x["recall"] for x in per_class])),
        "macro_f1": float(np.mean([x["f1"] for x in per_class])),
        "per_class": per_class,
        "confusion_matrix": cm,
    }

# data reading
@dataclass
class Sample:
    image_path: str
    image_name: str
    label: int
    label_name: str
    gt_w_128: float
    gt_h_128: float

def normalize_stem(path_or_name: str) -> str:
    return os.path.splitext(os.path.basename(path_or_name))[0]

def parse_meta_row_to_gt_wh_128(row):
    """
    parse according to the actual resize_meta.csv headers:
    image,src_w,src_h,new_w,new_h,scale,pad_left,pad_top,target_w,target_h,over_compress
    """
    required_keys = ["image", "new_w", "new_h"]
    for k in required_keys:
         if k not in row:
            raise KeyError(
                f"resize_meta.csv missing required field: {k}."
                f" current script requires header to contain at least: image,new_w,new_h"
            )

    gt_w_128 = float(row["new_w"])
    gt_h_128 = float(row["new_h"])
    return gt_w_128, gt_h_128


def build_meta_index(rows):
    """
    use the image field of csv to build an index.
    allow csv to have more entries than images, as long as each image can find its corresponding record in the index.
    """
    meta_index = {}
    for row in rows:
        if "image" not in row:
            continue
        raw_name = str(row["image"]).strip()
        if raw_name == "":
            continue
        stem = normalize_stem(raw_name)
        if stem not in meta_index:
            meta_index[stem] = row
    return meta_index


def load_samples_from_testset(testset_root: str) -> List[Sample]:
    samples: List[Sample] = []

    for class_name in sorted(os.listdir(testset_root)):
        class_dir = os.path.join(testset_root, class_name)
        if not os.path.isdir(class_dir):
            continue
        if class_name not in CLASS_TO_ID:
            print(f"[WARN] skip unknown class folder: {class_name}")
            continue

        label = CLASS_TO_ID[class_name]
        img_dir = os.path.join(class_dir, "img_instance")
        meta_csv = os.path.join(class_dir, "resize_meta.csv")

        if not os.path.isdir(img_dir):
            print(f"[WARN] missing img_instance: {img_dir}")
            continue
        if not os.path.isfile(meta_csv):
            print(f"[WARN] missing resize_meta.csv: {meta_csv}")
            continue

        with open(meta_csv, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                print(f"[WARN] empty csv: {meta_csv}")
                continue

            fieldnames = [str(x).strip() for x in reader.fieldnames]
            print(f"[INFO] loading {meta_csv}")
            print(f"[INFO] csv columns = {fieldnames}")

            if "image" not in fieldnames:
                raise KeyError(
                    f"{meta_csv} missing image field."
                    f" current resize_meta.csv must contain image,new_w,new_h"
                )

            rows = list(reader)
            meta_index = build_meta_index(rows)

        img_files = []
        for fn in sorted(os.listdir(img_dir)):
            full_path = os.path.join(img_dir, fn)
            if not os.path.isfile(full_path):
                continue
            _, ext = os.path.splitext(fn)
            if ext.lower() not in IMG_EXTS:
                continue
            img_files.append(full_path)

        total_imgs = len(img_files)
        matched_count = 0
        skipped_count = 0

        for image_path in img_files:
            image_name = os.path.basename(image_path)
            stem = normalize_stem(image_name)

            if stem not in meta_index:
                skipped_count += 1
                print(f"[WARN] no csv record for image: {image_name}")
                continue

            row = meta_index[stem]
            gt_w_128, gt_h_128 = parse_meta_row_to_gt_wh_128(row)

            samples.append(Sample(
                image_path=image_path,
                image_name=image_name,
                label=label,
                label_name=class_name,
                gt_w_128=float(gt_w_128),
                gt_h_128=float(gt_h_128),
            ))
            matched_count += 1

        print(
            f"[INFO] loaded {matched_count}/{total_imgs} images from {class_name} "
            f"(csv rows={len(rows)}, skipped images={skipped_count}, unused csv rows allowed)"
        )

    return samples

# main
@torch.no_grad()
def main():
    ensure_dir(OUT_DIR)
    vis_dir = os.path.join(OUT_DIR, "vis")
    sr_dir = os.path.join(OUT_DIR, "sr")
    ensure_dir(vis_dir)
    ensure_dir(sr_dir)

    device = torch.device(DEVICE if (DEVICE == "npu" and torch.npu.is_available()) else "cpu")
    print(f"[INFO] device = {device}")

    samples = load_samples_from_testset(TESTSET_ROOT)
    print(f"[INFO] loaded samples = {len(samples)}")
    if len(samples) == 0:
        raise RuntimeError("No samples found.")

    model_g = Generator(scale_factor=4).to(device).eval()
    model_h = Stage2Head(pretrained=True, num_classes=NUM_CLASSES).to(device).eval()

    ckpt = torch.load(CKPT_PATH, map_location=device)
    model_g.load_state_dict(ckpt["net_g"], strict=True)
    model_h.load_state_dict(ckpt["net_h"], strict=True)

    y_true, y_pred = [], []
    per_sample_rows = []

    abs_err_w_list, abs_err_h_list = [], []
    sq_err_w_list, sq_err_h_list = [], []
    nabs_err_w_list, nabs_err_h_list = [], []
    iou_list = []
    iou50_hits = 0

    for idx, sample in enumerate(samples):
        bgr = cv2.imread(sample.image_path, cv2.IMREAD_COLOR)
        if bgr is None:
            print(f"[WARN] unreadable image: {sample.image_path}")
            continue

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        if rgb.shape[0] != LR_SIZE or rgb.shape[1] != LR_SIZE:
            print(f"[WARN] image is not 32x32, got {rgb.shape[:2]} : {sample.image_path}")

        lr = to_tensor_01(rgb).unsqueeze(0).to(device)

        sr = model_g(lr).clamp(0, 1)
        cls_logits, box_refine = model_h(sr)

        prob = torch.softmax(cls_logits, dim=1)[0]
        pred_label = int(torch.argmax(prob).item())
        pred_prob = float(prob[pred_label].item())

        delta_w, delta_h = box_refine.squeeze(0).tolist()
        delta_w = float(delta_w)
        delta_h = float(delta_h)

        # target = [(128-w)/128, (128-h)/128]
        pred_w = float(SR_SIZE * (1.0 - delta_w))
        pred_h = float(SR_SIZE * (1.0 - delta_h))

        pred_w = clamp_float(pred_w, 1.0, SR_SIZE)
        pred_h = clamp_float(pred_h, 1.0, SR_SIZE)
        gt_w = clamp_float(sample.gt_w_128, 1.0, SR_SIZE)
        gt_h = clamp_float(sample.gt_h_128, 1.0, SR_SIZE)

        err_w = abs(pred_w - gt_w)
        err_h = abs(pred_h - gt_h)
        nerr_w = err_w / gt_w if gt_w > 1e-6 else 0.0
        nerr_h = err_h / gt_h if gt_h > 1e-6 else 0.0
        iou = center_box_iou(pred_w, pred_h, gt_w, gt_h)

        abs_err_w_list.append(err_w)
        abs_err_h_list.append(err_h)
        sq_err_w_list.append((pred_w - gt_w) ** 2)
        sq_err_h_list.append((pred_h - gt_h) ** 2)
        nabs_err_w_list.append(nerr_w)
        nabs_err_h_list.append(nerr_h)
        iou_list.append(iou)
        if iou >= 0.5:
            iou50_hits += 1

        y_true.append(sample.label)
        y_pred.append(pred_label)

        per_sample_rows.append({
            "image_name": sample.image_name,
            "gt_label": sample.label,
            "gt_label_name": sample.label_name,
            "pred_label": pred_label,
            "pred_label_name": ID_TO_CLASS.get(pred_label, str(pred_label)),
            "pred_prob": pred_prob,
            "gt_w_128": gt_w,
            "gt_h_128": gt_h,
            "pred_w_128": pred_w,
            "pred_h_128": pred_h,
            "abs_err_w": err_w,
            "abs_err_h": err_h,
            "norm_abs_err_w": nerr_w,
            "norm_abs_err_h": nerr_h,
            "center_iou": iou,
            "delta_w": delta_w,
            "delta_h": delta_h,
        })

        if SAVE_VIS and idx < MAX_VIS:
            sr_u8 = to_uint8_rgb(sr)
            vis = draw_center_box(sr_u8, gt_w, gt_h, color=(0, 255, 0), thickness=2)
            vis = draw_center_box(vis, pred_w, pred_h, color=(255, 0, 0), thickness=2)
            vis = put_text(vis, [
                f"GT: {sample.label_name}",
                f"Pred: {ID_TO_CLASS.get(pred_label, pred_label)}"
            ])
            save_rgb(os.path.join(vis_dir, sample.image_name), vis)
            save_rgb(os.path.join(sr_dir, sample.image_name), sr_u8)

        if (idx + 1) % 50 == 0 or (idx + 1) == len(samples):
            print(f"[INFO] processed {idx + 1}/{len(samples)}")

    cls_metrics = compute_prf1(y_true, y_pred, NUM_CLASSES)
    n = max(1, len(per_sample_rows))

    summary = {
        "num_samples": len(per_sample_rows),
        "classification": {
            "accuracy": cls_metrics["accuracy"],
            "macro_precision": cls_metrics["macro_precision"],
            "macro_recall": cls_metrics["macro_recall"],
            "macro_f1": cls_metrics["macro_f1"],
            "per_class": cls_metrics["per_class"],
        },
        "regression": {
            "mae_w_px": float(np.mean(abs_err_w_list)) if abs_err_w_list else 0.0,
            "mae_h_px": float(np.mean(abs_err_h_list)) if abs_err_h_list else 0.0,
            "rmse_w_px": float(math.sqrt(np.mean(sq_err_w_list))) if sq_err_w_list else 0.0,
            "rmse_h_px": float(math.sqrt(np.mean(sq_err_h_list))) if sq_err_h_list else 0.0,
            "norm_mae_w": float(np.mean(nabs_err_w_list)) if nabs_err_w_list else 0.0,
            "norm_mae_h": float(np.mean(nabs_err_h_list)) if nabs_err_h_list else 0.0,
            "mean_center_iou": float(np.mean(iou_list)) if iou_list else 0.0,
            "iou50_acc": float(iou50_hits / n),
        },
    }

    per_sample_csv = os.path.join(OUT_DIR, "per_sample_results.csv")
    if len(per_sample_rows) > 0:
        with open(per_sample_csv, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(per_sample_rows[0].keys()))
            writer.writeheader()
            writer.writerows(per_sample_rows)

    cm_csv = os.path.join(OUT_DIR, "confusion_matrix.csv")
    cm = cls_metrics["confusion_matrix"]
    with open(cm_csv, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["gt\\pred"] + [ID_TO_CLASS.get(i, str(i)) for i in range(NUM_CLASSES)])
        for i in range(NUM_CLASSES):
            writer.writerow([ID_TO_CLASS.get(i, str(i))] + cm[i].tolist())

    with open(os.path.join(OUT_DIR, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    with open(os.path.join(OUT_DIR, "summary.txt"), "w", encoding="utf-8") as f:
        f.write("=== Reconstruction Module Evaluation ===\n")
        f.write(f"num_samples: {summary['num_samples']}\n\n")

        f.write("[Classification]\n")
        f.write(f"accuracy        : {summary['classification']['accuracy']:.4f}\n")
        f.write(f"macro_precision : {summary['classification']['macro_precision']:.4f}\n")
        f.write(f"macro_recall    : {summary['classification']['macro_recall']:.4f}\n")
        f.write(f"macro_f1        : {summary['classification']['macro_f1']:.4f}\n\n")

        f.write("[Regression]\n")
        f.write(f"mae_w_px        : {summary['regression']['mae_w_px']:.4f}\n")
        f.write(f"mae_h_px        : {summary['regression']['mae_h_px']:.4f}\n")
        f.write(f"rmse_w_px       : {summary['regression']['rmse_w_px']:.4f}\n")
        f.write(f"rmse_h_px       : {summary['regression']['rmse_h_px']:.4f}\n")
        f.write(f"norm_mae_w      : {summary['regression']['norm_mae_w']:.4f}\n")
        f.write(f"norm_mae_h      : {summary['regression']['norm_mae_h']:.4f}\n")
        f.write(f"mean_center_iou : {summary['regression']['mean_center_iou']:.4f}\n")
        f.write(f"iou50_acc       : {summary['regression']['iou50_acc']:.4f}\n")

    print("\n========== SUMMARY ==========")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[OK] results saved to: {OUT_DIR}")
    if SAVE_VIS:
        print(f"[OK] vis dir: {vis_dir}")
        print(f"[OK] sr dir : {sr_dir}")


if __name__ == "__main__":
    main()
