import math
from dataclasses import dataclass

import numpy as np

from ..domain.scene import Scene
from .base import Effect
from .kit.compositing import blit
from .kit.sprites import emoji_sprite

MIDDLE_FINGER = "\U0001f595"
UPRIGHT, UPSIDE_DOWN = 0, 180


@dataclass
class Flyer:
    x: float
    y: float
    vy: float
    size: float
    angle: float
    phase: float

    @property
    def rising(self) -> bool:
        return self.vy < 0


class MiddleFingerEffect(Effect):
    PER_SECOND = 16
    OPENING_VOLLEY = 10

    def setup(self) -> None:
        self.sprite = emoji_sprite(MIDDLE_FINGER)
        self.flyers: list[Flyer] = []
        self.was_on = False

    def render(self, frame: np.ndarray, scene: Scene, on: bool, t: float, dt: float) -> np.ndarray:
        if on:
            count = self.rng.poisson(self.PER_SECOND * dt) if self.was_on else self.OPENING_VOLLEY
            for _ in range(count):
                self._launch()
        self.was_on = on
        self._fly(dt)
        if not self.flyers:
            return frame
        out = frame.copy()
        for f in self.flyers:
            x = f.x + 15 * self.unit * math.sin(2 * t + f.phase)
            angle = f.angle + 12 * math.sin(3 * t + f.phase)
            blit(out, self.sprite, (x, f.y), f.size / self.sprite.shape[0], angle)
        return out

    def _fly(self, dt: float) -> None:
        for f in self.flyers:
            f.y += f.vy * dt
        self.flyers = [f for f in self.flyers if (f.y > -f.size if f.rising else f.y < self.h + f.size)]

    def _launch(self) -> None:
        rng, u = self.rng, self.unit
        size = rng.uniform(60, 150) * u
        speed = rng.uniform(250, 550) * u
        from_below = rng.random() < 0.5
        staggered = size + rng.uniform(0, 150) * u
        y = self.h + staggered if from_below else -staggered
        x = rng.uniform(0.05, 0.95) * self.w
        velocity, angle = (-speed, UPRIGHT) if from_below else (speed, UPSIDE_DOWN)
        self.flyers.append(Flyer(x, y, velocity, size, angle, rng.uniform(0, 2 * math.pi)))
