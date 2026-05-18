import os
import cv2
from tqdm import tqdm

# config
HR_ROOT = ""   # input dir
LR_ROOT = ""   # output dir
SCALE = 4      # downsamle scale

def is_image_file(filename: str) -> bool:
    return filename.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"))

def modcrop(img, scale: int):
    h, w = img.shape[:2]
    h = h - (h % scale)
    w = w - (w % scale)
    if img.ndim == 2:
        return img[:h, :w]
    return img[:h, :w, :]
 
def bicubic_downsample(hr_bgr, scale: int):
    hr_bgr = modcrop(hr_bgr, scale)
    h, w = hr_bgr.shape[:2]
    lr_h, lr_w = h // scale, w // scale
    lr_bgr = cv2.resize(hr_bgr, (lr_w, lr_h), interpolation=cv2.INTER_CUBIC)
    return lr_bgr

def main():
    if not os.path.isdir(HR_ROOT):
        raise FileNotFoundError(f"HR_ROOT not found: {HR_ROOT}")

    hr_img_paths = []
    for root, _, files in os.walk(HR_ROOT):
        for f in files:
            if is_image_file(f):
                hr_img_paths.append(os.path.join(root, f))

    print("[INFO] LR generation ")
    print(f"[INFO] HR root: {HR_ROOT}")
    print(f"[INFO] LR root: {LR_ROOT}")
    print(f"[INFO] Scale factor: {SCALE}")
    print(f"[INFO] Number of HR images: {len(hr_img_paths)}")

    for hr_path in tqdm(hr_img_paths, desc="Generating LR crops"):
        rel_path = os.path.relpath(hr_path, HR_ROOT)
        lr_path = os.path.join(LR_ROOT, rel_path)

        os.makedirs(os.path.dirname(lr_path), exist_ok=True)

        hr_img = cv2.imread(hr_path, cv2.IMREAD_COLOR)  # BGR
        if hr_img is None:
            print(f"[WARNING] Failed to read: {hr_path}")
            continue

        lr_img = bicubic_downsample(hr_img, SCALE)  # BGR
        ok = cv2.imwrite(lr_path, lr_img)
        if not ok:
            print(f"[WARNING] Failed to write: {lr_path}")

    print("[INFO] Finished. LR crops saved to:", LR_ROOT)

if __name__ == "__main__":
    main()
