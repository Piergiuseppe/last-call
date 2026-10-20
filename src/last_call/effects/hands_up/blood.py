import math
from dataclasses import dataclass

import cv2
import numpy as np

from ..kit.compositing import layer_over, rotated
from ..kit.geometry import box_around, clipped_box, normalized, pixel
from ..kit.timing import ease_toward

BLOOD = (14, 8, 120)
BLOOD_OPAQUE = (*BLOOD, 255)
WET_SHINE = (45, 38, 190, 255)
BULLET_HOLE = (10, 6, 35, 255)
SPLAT_RADIUS = 40
SPLAT_STRETCHES = (1.0, 1.7, 2.6)
FOLLOW_PULL = 0.6


def splat_sprite(rng: np.random.Generator, radius: float, stretch: float, droplets: bool = True) -> np.ndarray:
    w, h = int(2 * radius * (1.3 + 1.6 * stretch)), int(3.2 * radius)
    sprite = np.zeros((h, w, 4), np.uint8)
    center = (w / 2, h / 2)
    _blot(sprite, rng, center, radius, stretch)
    if droplets:
        _droplets(sprite, rng, center, radius, stretch)
    return cv2.GaussianBlur(sprite, (0, 0), 1.0)


def _blot(sprite: np.ndarray, rng: np.random.Generator, center: tuple[float, float], radius: float, stretch: float) -> None:
    cx, cy = center
    for _ in range(5):
        axes = (int(radius * rng.uniform(0.55, 0.9) * (1 + 0.5 * (stretch - 1))), int(radius * rng.uniform(0.45, 0.8)))
        ox, oy = rng.normal(0, 0.18 * radius, 2)
        cv2.ellipse(sprite, (int(cx + ox), int(cy + oy)), axes, rng.uniform(-15, 15), 0, 360, BLOOD_OPAQUE, -1, cv2.LINE_AA)


def _droplets(sprite: np.ndarray, rng: np.random.Generator, center: tuple[float, float], radius: float, stretch: float) -> None:
    cx, cy = center
    for _ in range(int(rng.integers(6, 13))):
        angle = _mostly_along_flight(rng)
        reach = radius * rng.uniform(1.0, 1.0 + 1.5 * stretch)
        tip = (int(cx + math.cos(angle) * reach), int(cy + math.sin(angle) * reach))
        cv2.circle(sprite, tip, max(1, int(radius * rng.uniform(0.05, 0.17))), BLOOD_OPAQUE, -1, cv2.LINE_AA)
        if rng.random() < 0.4:
            start = (int(cx + math.cos(angle) * 0.6 * radius), int(cy + math.sin(angle) * 0.6 * radius))
            cv2.line(sprite, start, tip, BLOOD_OPAQUE, max(1, int(0.07 * radius)), cv2.LINE_AA)


def _mostly_along_flight(rng: np.random.Generator) -> float:
    return rng.normal(0, 0.45) if rng.random() < 0.7 else rng.uniform(-math.pi, math.pi)


def stamp_onto(layer: np.ndarray, sprite: np.ndarray, center: np.ndarray, scale: float, angle: float) -> bool:
    warped = rotated(sprite, angle, scale)
    size = warped.shape[0]
    lh, lw = layer.shape[:2]
    x0, y0 = int(center[0]) - size // 2, int(center[1]) - size // 2
    ox0, oy0, ox1, oy1 = clipped_box(x0, y0, x0 + size, y0 + size, lw, lh)
    if ox1 <= ox0 or oy1 <= oy0:
        return False
    layer_over(layer[oy0:oy1, ox0:ox1], warped[oy0 - y0 : oy1 - y0, ox0 - x0 : ox1 - x0])
    return True


@dataclass
class Drip:
    x: float
    y: float
    length: float
    speed: float
    width: float
    phase: float
    done: float = 0.0

    def point_at(self, run: float) -> tuple[int, int]:
        return int(self.x + 0.6 * self.width * math.sin(run / (6 * self.width) + self.phase)), int(self.y + run)

    def grow(self, canvas: np.ndarray, dt: float) -> None:
        if self.done >= self.length:
            return
        slowing_down = 1.2 - self.done / self.length
        run = min(self.length, self.done + self.speed * dt * slowing_down)
        tip = self.point_at(run)
        cv2.line(canvas, self.point_at(self.done), tip, BLOOD_OPAQUE, max(1, int(self.width)), cv2.LINE_AA)
        cv2.circle(canvas, tip, max(1, int(0.75 * self.width)), BLOOD_OPAQUE, -1, cv2.LINE_AA)
        self.done = run


class Drips:
    def __init__(self) -> None:
        self.items: list[Drip] = []

    def add(self, start: np.ndarray, length: float, speed: float, width: float, phase: float) -> None:
        self.items.append(Drip(start[0], start[1], length, speed, width, phase))

    def grow(self, canvas: np.ndarray, dt: float) -> None:
        for drip in self.items:
            drip.grow(canvas, dt)


class Wound:
    def __init__(self, size: int, origin: np.ndarray, up, scale: float) -> None:
        self.canvas = np.zeros((size, size, 4), np.uint8)
        self.center = np.array([size / 2, size / 2])
        self.reference = scale
        self.origin = np.asarray(origin, float)
        self.up = np.asarray(up, float)
        self.scale = float(scale)
        self.drips = Drips()

    def _to_screen(self) -> np.ndarray:
        up = self.up
        return np.array([[-up[1], -up[0]], [up[0], -up[1]]]) * (self.scale / self.reference)

    def local(self, point: np.ndarray) -> np.ndarray:
        return np.linalg.solve(self._to_screen(), np.asarray(point, float) - self.origin) + self.center

    def follow(self, origin: np.ndarray, up, scale: float) -> None:
        self.up = normalized(ease_toward(self.up, np.asarray(up, float), FOLLOW_PULL))
        self.origin = ease_toward(self.origin, np.asarray(origin, float), FOLLOW_PULL)
        self.scale = ease_toward(self.scale, scale, FOLLOW_PULL)

    def bleed(self, dt: float) -> None:
        self.drips.grow(self.canvas, dt)

    def paste(self, out: np.ndarray) -> None:
        h, w = out.shape[:2]
        matrix = self._to_screen()
        shift = self.origin - matrix @ self.center
        size = self.canvas.shape[0]
        corners = (matrix @ np.array([[0, size, 0, size], [0, 0, size, size]])).T + shift
        x0, y0, x1, y1 = box_around(corners, w, h)
        if x1 <= x0 or y1 <= y0:
            return
        affine = np.hstack([matrix, (shift - (x0, y0))[:, None]])
        warped = cv2.warpAffine(self.canvas, affine, (int(x1 - x0), int(y1 - y0)), flags=cv2.INTER_LINEAR)
        layer_over(out[y0:y1, x0:x1], warped)


def paint_entry_wound(wound: Wound, at: np.ndarray, rng: np.random.Generator) -> None:
    height = wound.reference
    _splashes_around(wound.canvas, at, height, rng)
    _clots(wound.canvas, at, height, rng)
    cv2.circle(wound.canvas, pixel(at), max(2, int(0.028 * height)), BULLET_HOLE, -1, cv2.LINE_AA)
    shine = (int(at[0] - 0.035 * height), int(at[1] - 0.03 * height))
    cv2.circle(wound.canvas, shine, max(1, int(0.008 * height)), WET_SHINE, -1, cv2.LINE_AA)
    _drips_toward_chin(wound, at, height, rng)


def _splashes_around(canvas: np.ndarray, at: np.ndarray, height: float, rng: np.random.Generator) -> None:
    for _ in range(int(rng.integers(18, 30))):
        angle, reach = rng.uniform(-math.pi, math.pi), height * rng.uniform(0.03, 0.3) ** 1.3 * 1.6
        spot = (int(at[0] + math.cos(angle) * reach), int(at[1] + math.sin(angle) * reach))
        cv2.circle(canvas, spot, max(1, int(height * rng.uniform(0.004, 0.018))), BLOOD_OPAQUE, -1, cv2.LINE_AA)


def _clots(canvas: np.ndarray, at: np.ndarray, height: float, rng: np.random.Generator) -> None:
    for _ in range(3):
        axes = (int(height * rng.uniform(0.04, 0.07)), int(height * rng.uniform(0.03, 0.05)))
        center = (int(at[0] + rng.normal(0, 0.015 * height)), int(at[1] + rng.normal(0, 0.015 * height)))
        cv2.ellipse(canvas, center, axes, rng.uniform(0, 180), 0, 360, BLOOD_OPAQUE, -1, cv2.LINE_AA)


def _drips_toward_chin(wound: Wound, at: np.ndarray, height: float, rng: np.random.Generator) -> None:
    for _ in range(4):
        start = at + (rng.normal(0, 0.035 * height), 0.03 * height)
        wound.drips.add(
            start,
            height * rng.uniform(0.3, 0.9),
            height * rng.uniform(0.25, 0.6),
            height * rng.uniform(0.008, 0.02),
            rng.uniform(0, 6),
        )
