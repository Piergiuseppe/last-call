from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from functools import cached_property

import numpy as np

from .landmarks import FACE_OVAL, LEFT_EYE, PALM, RIGHT_EYE, BodyPoint, FacePoint, Finger, HandPoint


class Side(StrEnum):
    LEFT = "left"
    RIGHT = "right"


@dataclass
class Hand:
    points: np.ndarray
    side: Side | None = None

    @property
    def wrist(self) -> np.ndarray:
        return self.points[HandPoint.WRIST]

    @property
    def thumb_tip(self) -> np.ndarray:
        return self.points[HandPoint.THUMB_TIP]

    @property
    def size(self) -> float:
        return float(np.linalg.norm(self.points[HandPoint.MIDDLE_MCP] - self.wrist))

    @property
    def palm(self) -> np.ndarray:
        return self.points[PALM].mean(axis=0)

    @property
    def direction(self) -> np.ndarray:
        return self.points[HandPoint.MIDDLE_MCP] - self.wrist

    def tip(self, finger: Finger) -> np.ndarray:
        return self.points[finger.tip]

    def reach(self, finger: Finger) -> float:
        wrist = self.wrist
        tip, joint = self.points[finger.tip], self.points[finger.middle_joint]
        return float(np.linalg.norm(tip - wrist) / (np.linalg.norm(joint - wrist) + 1e-6))


@dataclass
class Face:
    points: np.ndarray

    @property
    def oval(self) -> np.ndarray:
        return self.points[FACE_OVAL]

    @property
    def eyes(self) -> list[np.ndarray]:
        return [self.points[LEFT_EYE], self.points[RIGHT_EYE]]

    @property
    def forehead(self) -> np.ndarray:
        return self.points[FacePoint.FOREHEAD]

    @property
    def chin(self) -> np.ndarray:
        return self.points[FacePoint.CHIN]

    @property
    def center(self) -> np.ndarray:
        return self.oval.mean(axis=0)

    @property
    def height(self) -> float:
        return float(np.linalg.norm(self.forehead - self.chin))

    @property
    def width(self) -> float:
        return float(np.linalg.norm(self.points[FacePoint.RIGHT_EDGE] - self.points[FacePoint.LEFT_EDGE]))

    @property
    def up(self) -> np.ndarray:
        v = self.forehead - self.chin
        return v / (np.linalg.norm(v) + 1e-6)

    @property
    def crown(self) -> np.ndarray:
        return self.forehead + self.up * self.height * 0.35


@dataclass
class Body:
    points: np.ndarray
    visibility: np.ndarray

    def confidence(self, *joints: BodyPoint) -> float:
        return float(min(self.visibility[j] for j in joints))

    @property
    def shoulders(self) -> tuple[np.ndarray, np.ndarray]:
        return self.points[BodyPoint.LEFT_SHOULDER], self.points[BodyPoint.RIGHT_SHOULDER]

    @property
    def wrists(self) -> tuple[np.ndarray, np.ndarray]:
        return self.points[BodyPoint.LEFT_WRIST], self.points[BodyPoint.RIGHT_WRIST]

    @property
    def shoulders_confidence(self) -> float:
        return self.confidence(BodyPoint.LEFT_SHOULDER, BodyPoint.RIGHT_SHOULDER)

    @property
    def wrists_confidence(self) -> float:
        return self.confidence(BodyPoint.LEFT_WRIST, BodyPoint.RIGHT_WRIST)


@dataclass
class Oval:
    center: np.ndarray
    axes: np.ndarray
    angle: float


@dataclass
class Scene:
    width: int
    height: int
    hands: list[Hand]
    face: Face | None
    body: Body | None
    segment_person: Callable[[], np.ndarray] = field(repr=False)

    @cached_property
    def person_mask(self) -> np.ndarray:
        return self.segment_person()
