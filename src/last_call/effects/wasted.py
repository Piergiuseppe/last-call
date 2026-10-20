import math

import cv2
import numpy as np

from ..domain.landmarks import Finger
from ..domain.scene import Scene
from .base import OneShot
from .kit.color import grayscale
from .kit.compositing import blit
from .kit.sprites import premultiply, text_sprite
from .kit.timing import smoothstep

SHOTS = (0.0, 0.3, 0.6)
RED = (48, 48, 205)
TITLE_OUTLINE = (14, 14, 14)
SHATTERED_GLASS = (205, 205, 205)
HOLE, HOLE_RIM = (18, 18, 18), (150, 150, 150)
GREY_OUT = (1.0, 1.6)
BAND_IN = (1.3, 1.6)
TITLE_IN = (1.4, 1.65)
FADE_OUT = 0.7


def crack_sprite(rng: np.random.Generator, unit: float) -> np.ndarray:
    radius = int(rng.uniform(170, 240) * unit)
    size = 2 * radius + 1
    color = np.zeros((size, size, 3), np.uint8)
    alpha = np.zeros((size, size), np.uint8)
    center = np.array([radius, radius], float)
    rays, branches = _radial_cracks(rng, center, radius, unit)
    lines = rays + branches + _web_arcs(rng, rays, center, unit)
    for points in lines:
        _crack(color, alpha, points, 30, 110, max(2, int(3 * unit)))
    for points in lines:
        _crack(color, alpha, points, 238, 235, 1)
    _bullet_hole(color, alpha, radius, unit)
    return premultiply(np.dstack([color, alpha]))


def _heading(angle: float) -> np.ndarray:
    return np.array([math.cos(angle), math.sin(angle)])


def _radial_cracks(rng: np.random.Generator, center: np.ndarray, radius: int, unit: float) -> tuple[list, list]:
    rays, branches = [], []
    count = int(rng.integers(11, 17))
    start = rng.uniform(0, 2 * math.pi)
    for i in range(count):
        angle = start + 2 * math.pi * i / count + rng.uniform(-0.2, 0.2)
        length = rng.uniform(0.4, 0.98) * radius
        points, reached = [center + 7 * unit * _heading(angle)], 7 * unit
        while reached < length:
            step = rng.uniform(10, 26) * unit
            angle += rng.normal(0, 0.13)
            reached += step
            points.append(points[-1] + step * _heading(angle))
            if rng.random() < 0.12 and reached > 40 * unit:
                turn = angle + rng.choice([-1, 1]) * rng.uniform(0.4, 0.9)
                branch = rng.uniform(15, 50) * unit
                branches.append(np.array([points[-1], points[-1] + branch * _heading(turn)]))
        rays.append(np.array(points))
    return rays, branches


def _web_arcs(rng: np.random.Generator, rays: list, center: np.ndarray, unit: float) -> list:
    arcs = []
    for ring in (rng.uniform(22, 36), rng.uniform(55, 85), rng.uniform(105, 140)):
        ring *= unit
        for a, b in zip(rays, rays[1:] + rays[:1]):
            if rng.random() < 0.2:
                continue
            da, db = np.linalg.norm(a - center, axis=1), np.linalg.norm(b - center, axis=1)
            if da[-1] < ring or db[-1] < ring:
                continue
            p, q = a[np.argmax(da >= ring)], b[np.argmax(db >= ring)]
            middle = (p + q) / 2 + rng.normal(0, 3 * unit, 2)
            arcs.append(np.array([p, middle, q]))
    return arcs


def _crack(color: np.ndarray, alpha: np.ndarray, points: np.ndarray, shade: int, opacity: int, thickness: int) -> None:
    line = [np.round(points).astype(np.int32)]
    cv2.polylines(color, line, False, (shade,) * 3, thickness, cv2.LINE_AA)
    cv2.polylines(alpha, line, False, opacity, thickness, cv2.LINE_AA)


def _bullet_hole(color: np.ndarray, alpha: np.ndarray, radius: int, unit: float) -> None:
    center = (radius, radius)
    cv2.circle(color, center, int(17 * unit), SHATTERED_GLASS, -1, cv2.LINE_AA)
    cv2.circle(alpha, center, int(17 * unit), 120, -1, cv2.LINE_AA)
    cv2.circle(color, center, int(8 * unit), HOLE, -1, cv2.LINE_AA)
    cv2.circle(alpha, center, int(8 * unit), 255, -1, cv2.LINE_AA)
    cv2.circle(color, center, int(8 * unit), HOLE_RIM, 1, cv2.LINE_AA)


class WastedEffect(OneShot):
    DURATION = 5.0

    def setup(self) -> None:
        self.white = np.full((self.h, self.w, 3), 255, np.uint8)
        self.title = text_sprite("WASTED", "impact", int(124 * self.unit), RED, int(5 * self.unit), TITLE_OUTLINE)
        self.cracks: list[tuple[np.ndarray, np.ndarray]] = []

    def begin(self, scene: Scene) -> None:
        self.cracks = []

    def draw(self, frame: np.ndarray, scene: Scene, elapsed: float, dt: float) -> np.ndarray:
        while len(self.cracks) < len(SHOTS) and elapsed >= SHOTS[len(self.cracks)]:
            self._fire(scene)
        presence = 1 - smoothstep(self.DURATION - FADE_OUT, self.DURATION, elapsed)
        out = self._death_camera(frame, elapsed, presence)
        for sprite, center in self.cracks:
            blit(out, sprite, center, opacity=presence)
        self._draw_title(out, elapsed, presence)
        return out

    def _death_camera(self, frame: np.ndarray, elapsed: float, presence: float) -> np.ndarray:
        since = [elapsed - s for s in SHOTS[: len(self.cracks)]]
        kick = sum(math.exp(-d / 0.09) for d in since)
        flash = min(0.85, sum(0.85 * math.exp(-d / 0.05) for d in since))
        dead = smoothstep(*GREY_OUT, elapsed) * presence
        zoom = 1 + 0.08 * smoothstep(GREY_OUT[0], self.DURATION - FADE_OUT, elapsed) * presence
        out = frame
        if dead > 0:
            out = cv2.addWeighted(out, 1 - dead, grayscale(out), 0.88 * dead, 0)
        if kick > 0.02 or zoom > 1.0005:
            out = self._recoil(out, kick, zoom)
        if dead > 0:
            out = self._blur(out, dead)
        if flash > 0.02:
            out = cv2.addWeighted(out, 1 - flash, self.white, flash, 0)
        return frame.copy() if out is frame else out

    def _recoil(self, frame: np.ndarray, kick: float, zoom: float) -> np.ndarray:
        dx, dy = self.rng.normal(0, 24 * self.unit * kick, 2)
        matrix = cv2.getRotationMatrix2D((self.w / 2, self.h / 2), 0, zoom)
        matrix[:, 2] += dx, dy
        return cv2.warpAffine(frame, matrix, (self.w, self.h), borderMode=cv2.BORDER_REFLECT)

    def _blur(self, frame: np.ndarray, amount: float) -> np.ndarray:
        small = cv2.resize(frame, (self.w // 2, self.h // 2), interpolation=cv2.INTER_AREA)
        blurred = cv2.resize(cv2.GaussianBlur(small, (0, 0), 1.6 * self.unit), (self.w, self.h))
        return cv2.addWeighted(frame, 1 - 0.7 * amount, blurred, 0.7 * amount, 0)

    def _draw_title(self, out: np.ndarray, elapsed: float, presence: float) -> None:
        band = smoothstep(*BAND_IN, elapsed) * presence
        if band > 0:
            half = int(78 * self.unit)
            stripe = out[self.h // 2 - half : self.h // 2 + half]
            stripe[:] = (stripe * (1 - 0.62 * band)).astype(np.uint8)
        enter = smoothstep(*TITLE_IN, elapsed)
        if enter > 0:
            blit(out, self.title, (self.w / 2, self.h / 2), 1.3 - 0.3 * enter, opacity=enter * presence)

    def _fire(self, scene: Scene) -> None:
        target = self._near_first_shot() if self.cracks else self._first_target(scene)
        margin = 60 * self.unit
        target = np.clip(target, [margin, margin], [self.w - margin, self.h - margin])
        self.cracks.append((crack_sprite(self.rng, self.unit), target))

    def _first_target(self, scene: Scene) -> np.ndarray:
        if scene.hands:
            return scene.hands[0].tip(Finger.INDEX).copy()
        return np.array([self.w / 2, self.h / 2]) + self.rng.normal(0, 60 * self.unit, 2)

    def _near_first_shot(self) -> np.ndarray:
        return self.cracks[0][1] + self.rng.normal(0, 110 * self.unit, 2)
