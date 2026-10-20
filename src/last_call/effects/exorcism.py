import math

import cv2
import numpy as np

from ..domain.scene import Face, Scene
from .base import OneShot
from .kit.color import ColorGrade, desaturate
from .kit.compositing import caption, shake
from .kit.geometry import box_around, polygon_mask
from .kit.sprites import text_sprite
from .kit.timing import envelope, smoothstep

SUBTITLES = [
    "EXORCIZO TE, IMMUNDISSIME SPIRITUS",
    "VADE RETRO, SATANA",
    "SPIRITUS MORTADELLAE, EXI AB EO!",
    "DIABOLUS IN WIFI EST",
    "IN NOMINE PATRIS ET FILII ET SPIRITUS SANCTI",
]
SICK_GREEN = (0.62, 0.98, 0.78)
EYE_GROWTH = 1.55
RED_FLASH = 60


class ExorcismEffect(OneShot):
    DURATION = 8.0
    SUBTITLE_START, SUBTITLE_EVERY = 0.4, 1.5

    def setup(self) -> None:
        self.grade = ColorGrade.vignette(self.w, self.h, SICK_GREEN, brightness=1.0, depth=0.75, start=0.35, power=1.5)

    def draw(self, frame: np.ndarray, scene: Scene, elapsed: float, dt: float) -> np.ndarray:
        e = elapsed
        power = envelope(e, self.DURATION, 0.6, 1.0)
        out = self._possess(frame, power, e)
        if scene.face is not None:
            self._black_eyes(out, scene.face, power * smoothstep(0.3, 1.0, e))
            angle = self._head_turn(e)
            if abs(angle) > 0.5:
                self._rotate_head(out, scene.face, angle)
        out = shake(out, power * self.unit * (2 + 9 * smoothstep(1.0, 4.5, e)), self.rng)
        self._subtitle(out, e, power)
        return out

    def _possess(self, frame: np.ndarray, power: float, e: float) -> np.ndarray:
        out = desaturate(frame, 0.65 * power)
        out = self.grade.apply(out, power, self._dying_bulb(power))
        if power > 0.5 and math.sin(e * 23) > 0.97:
            out[..., 2] = cv2.add(out[..., 2], RED_FLASH)
        return out

    def _dying_bulb(self, power: float) -> float:
        flicker = 1 + self.rng.normal(0, 0.03)
        if self.rng.random() < 0.15 * power:
            flicker = self.rng.uniform(0.35, 0.75)
        return flicker

    def _black_eyes(self, out: np.ndarray, face: Face, opacity: float) -> None:
        if opacity <= 0:
            return
        for eye in face.eyes:
            self._blacken(out, eye, opacity)

    def _blacken(self, out: np.ndarray, eye: np.ndarray, opacity: float) -> None:
        center = eye.mean(axis=0)
        outline = center + (eye - center) * EYE_GROWTH
        box = box_around(outline, self.w, self.h, margin=6)
        x0, y0, x1, y1 = box
        if x1 <= x0 or y1 <= y0:
            return
        mask = cv2.GaussianBlur(polygon_mask(outline, box), (0, 0), max(1.0, 0.06 * (x1 - x0)))[..., None] * opacity
        roi = out[y0:y1, x0:x1]
        roi[:] = (roi * (1 - mask)).astype(np.uint8)
        self._wet_glint(out, center, x1 - x0, y1 - y0, opacity)

    @staticmethod
    def _wet_glint(out: np.ndarray, center: np.ndarray, width: int, height: int, opacity: float) -> None:
        size = max(1, int(0.05 * width))
        glint = (int(center[0] - 0.15 * width), int(center[1] - 0.12 * height))
        cv2.circle(out, glint, size, (int(70 * opacity),) * 3, -1, cv2.LINE_AA)

    def _head_turn(self, e: float) -> float:
        upside_down = 180 * smoothstep(1.5, 4.5, e)
        back_up = 180 * smoothstep(5.5, 7.0, e)
        angle = upside_down + back_up
        if 1.5 < e < 7.0:
            angle += self.rng.normal(0, 2.5)
        return angle % 360

    def _rotate_head(self, out: np.ndarray, face: Face, angle: float) -> None:
        radius = 0.85 * face.height
        cx, cy = face.center + face.up * 0.15 * face.height
        x0, y0 = int(max(0, cx - radius)), int(max(0, cy - radius))
        x1, y1 = int(min(self.w, cx + radius)), int(min(self.h, cy + radius))
        if x1 - x0 < 8 or y1 - y0 < 8:
            return
        roi = out[y0:y1, x0:x1]
        center = (cx - x0, cy - y0)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        turned = cv2.warpAffine(roi, matrix, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        mask = np.zeros(roi.shape[:2], np.float32)
        cv2.circle(mask, (int(center[0]), int(center[1])), int(radius * 0.88), 1.0, -1, cv2.LINE_AA)
        mask = cv2.GaussianBlur(mask, (0, 0), radius * 0.08)[..., None]
        roi[:] = (roi * (1 - mask) + turned * mask).astype(np.uint8)

    def _subtitle(self, out: np.ndarray, e: float, power: float) -> None:
        local = e - self.SUBTITLE_START
        index = int(local // self.SUBTITLE_EVERY)
        if local < 0 or index >= len(SUBTITLES):
            return
        within = local - index * self.SUBTITLE_EVERY
        opacity = power * envelope(within, self.SUBTITLE_EVERY, 0.15, 0.2)
        sprite = text_sprite(SUBTITLES[index], "serif_bold", int(40 * self.unit), (225, 235, 255), int(4 * self.unit), (0, 0, 40))
        caption(out, sprite, self.h - 70 * self.unit, opacity)
