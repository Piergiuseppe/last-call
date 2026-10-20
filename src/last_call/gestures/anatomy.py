import math
from dataclasses import dataclass

import numpy as np

from ..domain.landmarks import Finger, HandPoint
from ..domain.scene import Body, Face, Hand

EXTENDED_REACH = 1.15
CURLED_REACH = 1.0
LEFT_SIDE_WHEN_FACING_CAMERA = 1.0
SHOULDERS_SEEN = 0.5


def distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def mean_size(a: Hand, b: Hand) -> float:
    return (a.size + b.size) / 2


def extended(hand: Hand, finger: Finger) -> bool:
    return hand.reach(finger) > EXTENDED_REACH


def curled(hand: Hand, finger: Finger) -> bool:
    return hand.reach(finger) < CURLED_REACH


def open_hand(hand: Hand) -> bool:
    return all(extended(hand, finger) for finger in Finger)


def only_index_extended(hand: Hand) -> bool:
    return extended(hand, Finger.INDEX) and all(curled(hand, f) for f in (Finger.MIDDLE, Finger.RING, Finger.PINKY))


def points_up(hand: Hand, within: float) -> bool:
    dx, dy = hand.direction
    return -dy > math.cos(math.radians(within)) * math.hypot(dx, dy)


def lies_flat(hand: Hand, within: float) -> bool:
    dx, dy = hand.direction
    return abs(dx) > math.cos(math.radians(within)) * math.hypot(dx, dy)


def index_tilt(hand: Hand) -> float:
    dx, dy = hand.tip(Finger.INDEX) - hand.points[HandPoint.INDEX_MCP]
    return math.degrees(math.atan2(abs(dx), -dy))


def head_offset(face: Face, point: np.ndarray) -> tuple[float, float]:
    relative = point - face.center
    across = np.array([face.up[1], -face.up[0]])
    return float(relative @ across) / face.width, float(relative @ face.up) / face.height


def on_face(face: Face | None, point: np.ndarray) -> bool:
    if face is None:
        return False
    side, up = head_offset(face, point)
    return side**2 + ((up + 0.2) / 0.9) ** 2 < 1


def segments_cross(p1: np.ndarray, p2: np.ndarray, q1: np.ndarray, q2: np.ndarray) -> bool:
    def turn(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    return turn(p1, p2, q1) * turn(p1, p2, q2) < 0 and turn(q1, q2, p1) * turn(q1, q2, p2) < 0


@dataclass(frozen=True)
class Torso:
    middle: float
    width: float
    chin: float
    left_side: float


def torso_of(face: Face | None, body: Body | None) -> Torso | None:
    if body is not None and body.shoulders_confidence >= SHOULDERS_SEEN:
        left, right = body.shoulders
        width = distance(left, right)
        chin = face.chin[1] if face is not None else (left[1] + right[1]) / 2 - 0.3 * width
        return Torso((left[0] + right[0]) / 2, width, chin, float(np.sign(left[0] - right[0])))
    if face is not None:
        return Torso(face.center[0], 2.6 * face.width, face.chin[1], LEFT_SIDE_WHEN_FACING_CAMERA)
    return None
