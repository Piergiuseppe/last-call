import math

import cv2
import numpy as np


def rotated(sprite: np.ndarray, angle: float, scale: float) -> np.ndarray:
    sh, sw = sprite.shape[:2]
    size = int(math.hypot(sw, sh) * scale) + 2
    matrix = cv2.getRotationMatrix2D((sw / 2, sh / 2), angle, scale)
    matrix[:, 2] += (size - sw) / 2, (size - sh) / 2
    return cv2.warpAffine(sprite, matrix, (size, size), flags=cv2.INTER_LINEAR)


def blit(
    frame: np.ndarray,
    sprite: np.ndarray,
    center: tuple[float, float],
    scale: float = 1.0,
    angle: float = 0.0,
    opacity: float = 1.0,
) -> None:
    if opacity <= 0 or scale <= 0:
        return
    if angle == 0:
        sh, sw = sprite.shape[:2]
        w, h = max(1, int(sw * scale)), max(1, int(sh * scale))
        warped = cv2.resize(sprite, (w, h), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    else:
        warped = rotated(sprite, angle, scale)
        h, w = warped.shape[:2]
    fh, fw = frame.shape[:2]
    x0, y0 = int(center[0]) - w // 2, int(center[1]) - h // 2
    ox0, oy0, ox1, oy1 = max(0, x0), max(0, y0), min(fw, x0 + w), min(fh, y0 + h)
    if ox1 <= ox0 or oy1 <= oy0:
        return
    part = warped[oy0 - y0 : oy1 - y0, ox0 - x0 : ox1 - x0].astype(np.float32) * opacity
    roi = frame[oy0:oy1, ox0:ox1]
    roi[:] = np.clip(roi * (1 - part[..., 3:] / 255) + part[..., :3], 0, 255).astype(np.uint8)


def over(base: np.ndarray, color: np.ndarray, alpha: np.ndarray) -> None:
    base[:] = cv2.add(cv2.multiply(base, cv2.bitwise_not(alpha), scale=1 / 255), color)


def layer_over(base: np.ndarray, layer: np.ndarray) -> None:
    channels = base.shape[2]
    over(base, layer[..., :channels], cv2.merge([layer[..., 3]] * channels))


def caption(frame: np.ndarray, sprite: np.ndarray, y: float, opacity: float) -> None:
    width = frame.shape[1]
    blit(frame, sprite, (width / 2, y), min(1.0, 0.92 * width / sprite.shape[1]), opacity=opacity)


def radial_glow(radius: int, color: tuple[int, int, int], spread: float) -> np.ndarray:
    yy, xx = np.mgrid[-radius : radius + 1, -radius : radius + 1].astype(np.float32)
    falloff = np.exp(-(xx**2 + yy**2) / (2 * (radius / spread) ** 2))[..., None]
    return falloff * np.array(color, np.float32)


def add_glow(frame: np.ndarray, glow: np.ndarray, center: tuple[float, float], strength: float) -> None:
    r = glow.shape[0] // 2
    x, y = int(center[0]), int(center[1])
    fh, fw = frame.shape[:2]
    x0, y0, x1, y1 = max(0, x - r), max(0, y - r), min(fw, x + r + 1), min(fh, y + r + 1)
    if x1 <= x0 or y1 <= y0:
        return
    roi = frame[y0:y1, x0:x1]
    part = glow[y0 - (y - r) : y1 - (y - r), x0 - (x - r) : x1 - (x - r)]
    roi[:] = np.clip(roi + part * strength, 0, 255).astype(np.uint8)


def shake(frame: np.ndarray, amount: float, rng: np.random.Generator) -> np.ndarray:
    if amount < 0.5:
        return frame
    dx, dy = rng.normal(0, amount, 2)
    matrix = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(frame, matrix, frame.shape[1::-1], borderMode=cv2.BORDER_REFLECT)
