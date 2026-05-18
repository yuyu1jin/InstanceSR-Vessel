import cv2
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Literal

PadMode = Literal["sea", "fixed"]

@dataclass
class ResizePadMeta:
    scale: float
    pad_left: int
    pad_top: int
    new_w: int
    new_h: int
    src_w: int
    src_h: int
    target_w: int
    target_h: int

def estimate_sea_color(img: np.ndarray, border: int = 8) -> Tuple[int, int, int]:
    h, w = img.shape[:2]
    b = max(1, min(border, h // 10, w // 10))
    top = img[:b, :, :]
    bottom = img[-b:, :, :]
    left = img[:, :b, :]
    right = img[:, -b:, :]
    pixels = np.concatenate(
        [top.reshape(-1,3), bottom.reshape(-1,3), left.reshape(-1,3), right.reshape(-1,3)],
        axis=0
    )
    c = np.median(pixels, axis=0)
    return int(c[0]), int(c[1]), int(c[2])

class Letterbox:
    def __init__(
        self,
        target_w: int,
        target_h: int,
        pad_mode: PadMode = "sea",
        pad_color_fixed: Tuple[int,int,int] = (114,114,114),
        center: bool = True,
        allow_scale_up: bool = True,
        max_downscale: float = 6.0,  # beyond this scaling ratio, consider distortion too large (external decision to discard optional)
    ):
        self.target_w = int(target_w)
        self.target_h = int(target_h)
        self.pad_mode = pad_mode
        self.pad_color_fixed = pad_color_fixed
        self.center = center
        self.allow_scale_up = allow_scale_up
        self.max_downscale = float(max_downscale)

    def __call__(self, img_rgb: np.ndarray):
        assert img_rgb.ndim == 3 and img_rgb.shape[2] == 3
        src_h, src_w = img_rgb.shape[:2]

        scale = min(self.target_w / src_w, self.target_h / src_h)
        if not self.allow_scale_up:
            scale = min(scale, 1.0)

        new_w = int(round(src_w * scale))
        new_h = int(round(src_h * scale))

        # interpolation: use area for downscaling, use linear for upscaling
        interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        resized = cv2.resize(img_rgb, (new_w, new_h), interpolation=interp)

        pad_w = self.target_w - new_w
        pad_h = self.target_h - new_h

        if self.center:
            pad_left = pad_w // 2
            pad_right = pad_w - pad_left
            pad_top = pad_h // 2
            pad_bottom = pad_h - pad_top
        else:
            pad_left, pad_top = 0, 0
            pad_right, pad_bottom = pad_w, pad_h

        if self.pad_mode == "sea":
            pad_color = estimate_sea_color(img_rgb)
        else:
            pad_color = self.pad_color_fixed

        out = cv2.copyMakeBorder(
            resized,
            pad_top, pad_bottom, pad_left, pad_right,
            borderType=cv2.BORDER_CONSTANT,
            value=pad_color
        )

        meta = ResizePadMeta(
            scale=scale,
            pad_left=pad_left,
            pad_top=pad_top,
            new_w=new_w,
            new_h=new_h,
            src_w=src_w,
            src_h=src_h,
            target_w=self.target_w,
            target_h=self.target_h
        )

        # over-compress judgment: scale < 1/max_downscale
        over_compress = (meta.scale < (1.0 / self.max_downscale))

        return out, meta, over_compress