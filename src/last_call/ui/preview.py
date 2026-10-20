from collections.abc import Sequence
from collections.abc import Set as AbstractSet
from typing import Self

import cv2
import numpy as np

from ..domain.gesture import Gesture
from ..domain.landmarks import BodyPoint
from ..domain.scene import Scene, Side
from ..gestures.detectors import squirtle_oval
from .catalog import Entry

WINDOW = "last-call"
QUIT_KEYS = (ord("q"), 27)
HANDS, FACE, BODY, OVAL = (0, 255, 0), (255, 200, 0), (255, 0, 255), (0, 255, 255)
BODY_JOINTS = (BodyPoint.LEFT_SHOULDER, BodyPoint.RIGHT_SHOULDER, BodyPoint.LEFT_WRIST, BodyPoint.RIGHT_WRIST)


class PreviewClosed(Exception):
    pass


def draw_landmarks(view: np.ndarray, scene: Scene) -> None:
    for hand in scene.hands:
        for x, y in hand.points.astype(int):
            cv2.circle(view, (x, y), 3, HANDS, -1)
        if hand.side is not None:
            x, y = hand.wrist.astype(int)
            label = "S" if hand.side == Side.LEFT else "D"
            cv2.putText(view, label, (x - 10, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, HANDS, 2)
    if scene.face is not None:
        cv2.polylines(view, [scene.face.oval.astype(np.int32)], True, FACE, 1)
    if scene.body is not None:
        for joint in BODY_JOINTS:
            if scene.body.visibility[joint] > 0.2:
                cv2.circle(view, tuple(scene.body.points[joint].astype(int)), 6, BODY, 2)
    oval = squirtle_oval(scene.hands)
    if oval is not None:
        cv2.ellipse(view, tuple(oval.center.astype(int)), tuple(oval.axes.astype(int)), oval.angle, 0, 360, OVAL, 2)


def draw_status(view: np.ndarray, scene: Scene, gesture: Gesture | None, active: AbstractSet[Gesture], fps: float) -> None:
    lines = [
        f"{fps:.0f} fps  mani: {len(scene.hands)}",
        f"gesto: {gesture or '-'}",
        f"attivi: {', '.join(sorted(active)) or '-'}",
    ]
    for i, line in enumerate(lines):
        cv2.putText(view, line, (20, 40 + 32 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.8, HANDS, 2)


class Preview:
    def __init__(self, entries: Sequence[Entry], debug: bool, mirror: bool) -> None:
        self.keys = {ord(entry.key): entry.gesture for entry in entries}
        self.debug, self.mirror = debug, mirror

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_) -> None:
        cv2.destroyAllWindows()

    def show(self, frame: np.ndarray, scene: Scene, gesture: Gesture | None, active: AbstractSet[Gesture], fps: float) -> None:
        view = frame.copy()
        if self.debug:
            draw_landmarks(view, scene)
        if self.mirror:
            view = cv2.flip(view, 1)
        if self.debug:
            draw_status(view, scene, gesture, active, fps)
        cv2.imshow(WINDOW, view)

    def requested(self) -> Gesture | None:
        key = cv2.waitKey(1) & 0xFF
        if key in QUIT_KEYS:
            raise PreviewClosed
        return self.keys.get(key)
