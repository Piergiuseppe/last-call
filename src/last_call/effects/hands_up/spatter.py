import math

import cv2
import numpy as np

from ..kit.compositing import layer_over
from ..kit.geometry import pixel
from .blood import BLOOD, SPLAT_RADIUS, SPLAT_STRETCHES, Drips, stamp_onto

GRAVITY = 1800
MAX_SPLAT_SCALE = 2.0
VISIBLE_TAIL = 14


class LensSpatter:
    def __init__(self, width: int, height: int, unit: float, rng: np.random.Generator, splats: dict) -> None:
        self.w, self.h, self.unit, self.rng = width, height, unit, rng
        self.splats = splats
        self.pos, self.vel = np.zeros((0, 2)), np.zeros((0, 2))
        self.size, self.lands_at = np.zeros(0), np.zeros(0)
        self.lens = np.zeros((height, width, 4), np.uint8)
        self.drips = Drips()
        self.stained = False

    def burst(self, origin: np.ndarray, direction: np.ndarray, count: int, now: float) -> None:
        rng, u = self.rng, self.unit
        heading = math.atan2(direction[1], direction[0])
        backwards = rng.random(count) < 0.15
        angle = np.where(backwards, heading + math.pi + rng.normal(0, 0.5, count), heading + rng.normal(0, 0.55, count))
        speed = rng.uniform(300, 1500, count) * u
        self.pos = np.vstack([self.pos, origin + rng.normal(0, 4 * u, (count, 2))])
        self.vel = np.vstack([self.vel, np.stack([np.cos(angle), np.sin(angle)], axis=1) * speed[:, None]])
        self.size = np.concatenate([self.size, rng.uniform(1.5, 6.5, count) * u])
        toward_lens = rng.random(count) < 0.45
        self.lands_at = np.concatenate([self.lands_at, np.where(toward_lens, now + rng.uniform(0.04, 0.5, count), np.inf)])

    def fly(self, out: np.ndarray, now: float, dt: float) -> None:
        if not len(self.pos):
            return
        previous = self.pos.copy()
        self.vel[:, 1] += GRAVITY * self.unit * dt
        self.pos = self.pos + self.vel * dt
        landed = self.lands_at <= now
        for i in np.flatnonzero(landed):
            self.land(self.pos[i], self.vel[i], self.size[i])
        margin = 50 * self.unit
        inside = (self.pos[:, 0] > -margin) & (self.pos[:, 0] < self.w + margin) & (self.pos[:, 1] < self.h + margin)
        flying = ~landed & inside
        self._draw_short_tails(out, previous[flying], self.pos[flying], self.size[flying])
        self.pos, self.vel = self.pos[flying], self.vel[flying]
        self.size, self.lands_at = self.size[flying], self.lands_at[flying]

    def _draw_short_tails(self, out: np.ndarray, starts: np.ndarray, ends: np.ndarray, sizes: np.ndarray) -> None:
        tail = VISIBLE_TAIL * self.unit
        for start, end, size in zip(starts, ends, sizes):
            step = end - start
            start = end - step * min(1.0, tail / (np.linalg.norm(step) + 1e-6))
            cv2.line(out, pixel(start), pixel(end), BLOOD, max(1, int(0.7 * size)), cv2.LINE_AA)

    def land(self, point: np.ndarray, velocity: np.ndarray, size: float) -> None:
        rng, u = self.rng, self.unit
        if not (0 <= point[0] < self.w and 0 <= point[1] < self.h):
            return
        sprites = self.splats[self._stretch(np.linalg.norm(velocity))]
        scale = min(MAX_SPLAT_SCALE, size * rng.uniform(2.5, 5.0) / (SPLAT_RADIUS * u))
        on_screen_angle = math.degrees(math.atan2(-velocity[1], velocity[0]))
        if not stamp_onto(self.lens, sprites[int(rng.integers(len(sprites)))], point, scale, on_screen_angle):
            return
        self.stained = True
        if scale > 0.45 and rng.random() < 0.6:
            self._drip_down_the_lens(point, SPLAT_RADIUS * u * scale)

    def _stretch(self, speed: float) -> float:
        slow, medium, fast = SPLAT_STRETCHES
        return slow if speed < 600 * self.unit else (medium if speed < 1100 * self.unit else fast)

    def _drip_down_the_lens(self, point: np.ndarray, radius: float) -> None:
        rng, u = self.rng, self.unit
        start = point + (rng.normal(0, 0.2 * radius), 0.4 * radius)
        self.drips.add(start, rng.uniform(60, 260) * u, rng.uniform(30, 90) * u, max(2.0, 0.22 * radius), rng.uniform(0, 6))

    def cover(self, out: np.ndarray, dt: float) -> None:
        self.drips.grow(self.lens, dt)
        if self.stained:
            layer_over(out, self.lens)
