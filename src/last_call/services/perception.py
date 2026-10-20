from typing import Self

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

from ..domain.scene import Body, Face, Hand, Scene, Side
from .models import FACE_MODEL, HAND_MODEL, POSE_MODEL, SEGMENTATION_MODEL, model_path

ANALYSIS_SCALE = 2
POSE_EVERY = 2
HAND_CONFIDENCE = 0.4
GHOST_HAND_RATIO = 0.3
MASK_BLUR = 3
VIDEO = vision.RunningMode.VIDEO


def _model(name: str) -> BaseOptions:
    return BaseOptions(model_asset_path=model_path(name))


def _ghost(hand: Hand, face: Face) -> bool:
    return hand.size <= GHOST_HAND_RATIO * face.height


class PerceptionService:
    def __init__(self, width: int, height: int) -> None:
        self.width, self.height = width, height
        self.scale = np.array([width, height])
        self.hands = vision.HandLandmarker.create_from_options(
            vision.HandLandmarkerOptions(
                base_options=_model(HAND_MODEL),
                running_mode=VIDEO,
                num_hands=2,
                min_hand_detection_confidence=HAND_CONFIDENCE,
                min_hand_presence_confidence=HAND_CONFIDENCE,
                min_tracking_confidence=HAND_CONFIDENCE,
            )
        )
        self.faces = vision.FaceLandmarker.create_from_options(
            vision.FaceLandmarkerOptions(base_options=_model(FACE_MODEL), running_mode=VIDEO)
        )
        self.poses = vision.PoseLandmarker.create_from_options(
            vision.PoseLandmarkerOptions(base_options=_model(POSE_MODEL), running_mode=VIDEO)
        )
        self.segmenter = vision.ImageSegmenter.create_from_options(
            vision.ImageSegmenterOptions(base_options=_model(SEGMENTATION_MODEL), running_mode=VIDEO)
        )
        self.last_ms = -1
        self.frames = 0
        self.body: Body | None = None

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_) -> None:
        for task in (self.hands, self.faces, self.poses, self.segmenter):
            task.close()

    def perceive(self, frame: np.ndarray, t: float) -> Scene:
        ms = self._timestamp(t)
        small = cv2.resize(frame, (self.width // ANALYSIS_SCALE, self.height // ANALYSIS_SCALE), interpolation=cv2.INTER_AREA)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
        hands = self._find_hands(image, ms)
        face = self._find_face(image, ms)
        self._track_body(image, ms)
        if face is not None:
            hands = [hand for hand in hands if not _ghost(hand, face)]
        return Scene(self.width, self.height, hands, face, self.body, lambda: self._segment(image, ms, small.shape[:2]))

    def _timestamp(self, t: float) -> int:
        self.last_ms = max(int(t * 1000), self.last_ms + 1)
        return self.last_ms

    def _pixels(self, landmarks) -> np.ndarray:
        return np.array([[p.x, p.y] for p in landmarks]) * self.scale

    def _find_hands(self, image: mp.Image, ms: int) -> list[Hand]:
        found = self.hands.detect_for_video(image, ms)
        return [
            Hand(self._pixels(points), Side(labels[0].category_name.lower()))
            for points, labels in zip(found.hand_landmarks, found.handedness)
        ]

    def _find_face(self, image: mp.Image, ms: int) -> Face | None:
        faces = self.faces.detect_for_video(image, ms).face_landmarks
        return Face(self._pixels(faces[0])) if faces else None

    def _track_body(self, image: mp.Image, ms: int) -> None:
        self.frames += 1
        if self.frames % POSE_EVERY != 1:
            return
        poses = self.poses.detect_for_video(image, ms).pose_landmarks
        if not poses:
            self.body = None
            return
        self.body = Body(self._pixels(poses[0]), np.array([p.visibility or 0.0 for p in poses[0]]))

    def _segment(self, image: mp.Image, ms: int, shape: tuple[int, int]) -> np.ndarray:
        person = self.segmenter.segment_for_video(image, ms).confidence_masks[0].numpy_view().reshape(shape)
        return cv2.GaussianBlur(cv2.resize(person, (self.width, self.height)), (0, 0), MASK_BLUR)
