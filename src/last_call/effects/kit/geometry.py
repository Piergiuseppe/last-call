import math

import cv2
import numpy as np


def pixel(point: np.ndarray) -> tuple[int, int]:
    return int(point[0]), int(point[1])


def normalized(vector: np.ndarray) -> np.ndarray:
    return vector / (np.linalg.norm(vector) + 1e-6)


def degrees_of(vector: np.ndarray) -> float:
    return math.degrees(math.atan2(vector[1], vector[0]))


def clipped_box(x0: int, y0: int, x1: int, y1: int, width: int, height: int) -> tuple[int, int, int, int]:
    return max(0, x0), max(0, y0), min(width, x1), min(height, y1)


def box_around(points: np.ndarray, width: int, height: int, margin: int = 0) -> tuple[int, int, int, int]:
    x0, y0 = np.floor(points.min(axis=0)).astype(int) - margin
    x1, y1 = np.ceil(points.max(axis=0)).astype(int) + margin
    return clipped_box(x0, y0, x1, y1, width, height)


def polygon_mask(outline: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    x0, y0, x1, y1 = box
    mask = np.zeros((y1 - y0, x1 - x0), np.float32)
    cv2.fillPoly(mask, [np.round(outline - (x0, y0)).astype(np.int32)], 1.0, cv2.LINE_AA)
    return mask
