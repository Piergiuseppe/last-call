from collections.abc import Callable
from dataclasses import dataclass

from ..domain.gesture import Gesture
from ..domain.scene import Scene
from . import detectors


@dataclass(frozen=True)
class Rule:
    gesture: Gesture
    detect: Callable[[Scene], bool]
    hold: float
    grace: float


RULES = (
    Rule(Gesture.GRAVE, detectors.nobody_in_frame, hold=3.0, grace=0.2),
    Rule(Gesture.SQUIRTLE, detectors.oval_between_hands, hold=0.35, grace=0.25),
    Rule(Gesture.BEATIFICATION, detectors.praying, hold=0.35, grace=0.25),
    Rule(Gesture.CREMATION, detectors.flame, hold=0.4, grace=0.25),
    Rule(Gesture.EXORCISM, detectors.crossed_index_fingers, hold=0.35, grace=0.25),
    Rule(Gesture.ELECTRIC_CHAIR, detectors.hands_on_head, hold=0.5, grace=0.3),
    Rule(Gesture.HANDS_UP, detectors.surrender, hold=0.5, grace=0.3),
    Rule(Gesture.HEART_MONITOR, detectors.pulse_check, hold=0.4, grace=0.3),
    Rule(Gesture.REAPER, detectors.time_out, hold=0.4, grace=0.3),
    Rule(Gesture.WASTED, detectors.finger_gun, hold=0.35, grace=0.2),
    Rule(Gesture.HAND_OF_GOD, detectors.pointing_at_the_sky, hold=0.45, grace=0.2),
    Rule(Gesture.MIDDLE_FINGER, detectors.middle_finger, hold=0.25, grace=0.3),
    Rule(Gesture.FUNERAL, detectors.crossed_arms, hold=0.5, grace=0.3),
    Rule(Gesture.TESTAMENT, detectors.hand_on_heart, hold=0.7, grace=0.3),
)
