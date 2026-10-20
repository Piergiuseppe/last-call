import cv2
import numpy as np


def grayscale(frame: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)


def desaturate(frame: np.ndarray, amount: float) -> np.ndarray:
    return cv2.addWeighted(frame, 1 - amount, grayscale(frame), amount, 0)


def crossfade(original: np.ndarray, edited: np.ndarray, amount: float) -> np.ndarray:
    return edited if amount >= 1 else cv2.addWeighted(edited, amount, original, 1 - amount, 0)


def radial_falloff(width: int, height: int, start: float, spread: float, power: float) -> np.ndarray:
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    distance = np.hypot((xx - width / 2) / (width / 2), (yy - height / 2) / (height / 2))
    return np.clip((distance - start) / spread, 0, 1) ** power


class ColorGrade:
    def __init__(self, gains: np.ndarray) -> None:
        self.gains = gains
        self.neutral = np.full_like(gains, 255)

    @classmethod
    def vignette(cls, width: int, height: int, tint, brightness: float, depth: float, start: float, power: float):
        falloff = radial_falloff(width, height, start, 0.9, power)
        light = brightness * (1 - depth * falloff)
        return cls(np.clip(light[..., None] * np.array(tint, np.float32) * 255, 0, 255).astype(np.uint8))

    def apply(self, frame: np.ndarray, strength: float, flicker: float = 1.0) -> np.ndarray:
        gains = cv2.addWeighted(self.gains, strength * flicker, self.neutral, 1 - strength, 0)
        return cv2.multiply(frame, gains, scale=1 / 255)
