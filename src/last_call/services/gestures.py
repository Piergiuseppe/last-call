from dataclasses import dataclass

from ..domain.gesture import Gesture
from ..domain.scene import Scene
from ..gestures.hold import Hold
from ..gestures.rules import RULES, Rule


@dataclass(frozen=True)
class Recognition:
    gesture: Gesture | None
    active: frozenset[Gesture]


class GestureService:
    def __init__(self, rules: tuple[Rule, ...] = RULES) -> None:
        self.rules = rules
        self.holds = {rule.gesture: Hold(rule.hold, rule.grace) for rule in rules}

    def recognize(self, scene: Scene) -> Gesture | None:
        return next((rule.gesture for rule in self.rules if rule.detect(scene)), None)

    def update(self, scene: Scene, t: float) -> Recognition:
        gesture = self.recognize(scene)
        active = frozenset(g for g, hold in self.holds.items() if hold.update(g == gesture, t))
        return Recognition(gesture, active)
