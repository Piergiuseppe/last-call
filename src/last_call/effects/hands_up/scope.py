import math

import cv2
import numpy as np

RETICLE = (10, 10, 10)
RIM = (20, 20, 20)
CROSSHAIR = (40, 40, 255)
MASK_SCALE = 4


def look_through_scope(frame: np.ndarray, center: np.ndarray, radius: float) -> np.ndarray:
    h, w = frame.shape[:2]
    cx, cy = center
    mask = np.zeros((h // MASK_SCALE, w // MASK_SCALE), np.uint8)
    cv2.circle(mask, (int(cx / MASK_SCALE), int(cy / MASK_SCALE)), int(radius / MASK_SCALE), 255, -1, cv2.LINE_AA)
    mask = cv2.resize(cv2.GaussianBlur(mask, (0, 0), 1.5), (w, h), interpolation=cv2.INTER_LINEAR)
    return cv2.multiply(frame, cv2.merge([mask, mask, mask]), scale=1 / 255)


def draw_reticle(out: np.ndarray, center: np.ndarray, radius: float, unit: float) -> None:
    cx, cy = center
    thick = max(1, int(2 * unit))
    cv2.circle(out, (int(cx), int(cy)), int(radius), RIM, max(2, int(6 * unit)), cv2.LINE_AA)
    gap = int(0.12 * radius)
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        _draw_arm(out, (cx, cy), (dx, dy), gap, radius, thick, unit)
    cv2.line(out, (int(cx - gap / 2), int(cy)), (int(cx + gap / 2), int(cy)), CROSSHAIR, thick, cv2.LINE_AA)
    cv2.line(out, (int(cx), int(cy - gap / 2)), (int(cx), int(cy + gap / 2)), CROSSHAIR, thick, cv2.LINE_AA)


def _draw_arm(out: np.ndarray, center, direction, gap: int, radius: float, thick: int, unit: float) -> None:
    (cx, cy), (dx, dy) = center, direction
    inner = (int(cx + dx * gap), int(cy + dy * gap))
    outer = (int(cx + dx * radius), int(cy + dy * radius))
    thickening = (int(cx + dx * 0.6 * radius), int(cy + dy * 0.6 * radius))
    cv2.line(out, inner, outer, RETICLE, thick, cv2.LINE_AA)
    cv2.line(out, thickening, outer, RETICLE, 4 * thick, cv2.LINE_AA)
    for k in range(1, 6):
        tick = (cx + dx * gap * 1.6 * k, cy + dy * gap * 1.6 * k)
        if math.hypot(tick[0] - cx, tick[1] - cy) > radius * 0.95:
            break
        half = 7 * unit if k % 2 else 12 * unit
        a = (int(tick[0] - dy * half), int(tick[1] - dx * half))
        b = (int(tick[0] + dy * half), int(tick[1] + dx * half))
        cv2.line(out, a, b, RETICLE, thick, cv2.LINE_AA)
