import cv2
import numpy as np


def smooth_noise(rng: np.random.Generator, shape: tuple[int, int], cell: int) -> np.ndarray:
    h, w = shape
    coarse = rng.normal(0, 1, (h // max(1, cell) + 2, w // max(1, cell) + 2)).astype(np.float32)
    return cv2.resize(coarse, (w, h), interpolation=cv2.INTER_CUBIC)
