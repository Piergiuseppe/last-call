from collections.abc import Set as AbstractSet

import numpy as np

from ..domain.gesture import Gesture
from ..domain.scene import Scene
from ..effects.layers import LAYERS

FORCED_SECONDS = 1.5


class EffectService:
    def __init__(self, width: int, height: int, name: str) -> None:
        self.effects = {gesture: effect(width, height, name) for gesture, effect in LAYERS.items()}
        self.forced_until = dict.fromkeys(self.effects, -1.0)

    def force(self, gesture: Gesture, t: float) -> None:
        self.forced_until[gesture] = t + FORCED_SECONDS

    def active(self, recognized: AbstractSet[Gesture], t: float) -> frozenset[Gesture]:
        return frozenset(g for g in self.effects if g in recognized or t < self.forced_until[g])

    def render(self, frame: np.ndarray, scene: Scene, active: AbstractSet[Gesture], t: float, dt: float) -> np.ndarray:
        out = frame
        for gesture, effect in self.effects.items():
            out = effect.render(out, scene, gesture in active, t, dt)
        return out
