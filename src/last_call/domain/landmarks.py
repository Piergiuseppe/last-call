from collections import defaultdict
from enum import Enum, IntEnum

from mediapipe.tasks.python.vision import FaceLandmarksConnections


class HandPoint(IntEnum):
    WRIST = 0
    THUMB_CMC = 1
    THUMB_MCP = 2
    THUMB_IP = 3
    THUMB_TIP = 4
    INDEX_MCP = 5
    INDEX_PIP = 6
    INDEX_DIP = 7
    INDEX_TIP = 8
    MIDDLE_MCP = 9
    MIDDLE_PIP = 10
    MIDDLE_DIP = 11
    MIDDLE_TIP = 12
    RING_MCP = 13
    RING_PIP = 14
    RING_DIP = 15
    RING_TIP = 16
    PINKY_MCP = 17
    PINKY_PIP = 18
    PINKY_DIP = 19
    PINKY_TIP = 20


class Finger(Enum):
    INDEX = (HandPoint.INDEX_TIP, HandPoint.INDEX_PIP)
    MIDDLE = (HandPoint.MIDDLE_TIP, HandPoint.MIDDLE_PIP)
    RING = (HandPoint.RING_TIP, HandPoint.RING_PIP)
    PINKY = (HandPoint.PINKY_TIP, HandPoint.PINKY_PIP)

    @property
    def tip(self) -> HandPoint:
        return self.value[0]

    @property
    def middle_joint(self) -> HandPoint:
        return self.value[1]


PALM = [HandPoint.WRIST, HandPoint.INDEX_MCP, HandPoint.MIDDLE_MCP, HandPoint.RING_MCP, HandPoint.PINKY_MCP]
THUMB = [HandPoint.THUMB_MCP, HandPoint.THUMB_IP, HandPoint.THUMB_TIP]
INDEX = [HandPoint.INDEX_MCP, HandPoint.INDEX_PIP, HandPoint.INDEX_DIP, HandPoint.INDEX_TIP]


class BodyPoint(IntEnum):
    NOSE = 0
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16


class FacePoint(IntEnum):
    FOREHEAD = 10
    CHIN = 152
    RIGHT_EDGE = 234
    LEFT_EDGE = 454


def _contour(connections) -> list[int]:
    neighbours = defaultdict(list)
    for c in connections:
        neighbours[c.start].append(c.end)
        neighbours[c.end].append(c.start)
    order = [connections[0].start]
    while unvisited := [n for n in neighbours[order[-1]] if n not in order]:
        order.append(unvisited[0])
    return order


FACE_OVAL = _contour(FaceLandmarksConnections.FACE_LANDMARKS_FACE_OVAL)
LEFT_EYE = _contour(FaceLandmarksConnections.FACE_LANDMARKS_LEFT_EYE)
RIGHT_EYE = _contour(FaceLandmarksConnections.FACE_LANDMARKS_RIGHT_EYE)
