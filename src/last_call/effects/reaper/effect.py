import math
from dataclasses import dataclass

import cv2
import numpy as np

from ...domain.scene import Scene
from ..base import OneShot
from ..kit.color import ColorGrade, desaturate
from ..kit.compositing import blit, caption, over, rotated
from ..kit.geometry import clipped_box
from ..kit.sprites import emoji_sprite, text_sprite
from ..kit.timing import ease_toward, envelope, smoothstep
from .props import CLOTH, CLOTH_HIGHLIGHT, bony_hand, cloak, scythe, skull_in_shadow

COLD = (1.0, 0.86, 0.76)
SUBTITLE, SUBTITLE_OUTLINE = (225, 235, 245), (10, 0, 0)
HOURGLASS, SCROLL = "⏳", "\U0001f4dc"
SPRITE_FACE_HEIGHT = 200.0
LARGER_THAN_YOU = 1.3
SUBTITLES = [
    (1.2, 3.0, "È ora, {nome}."),
    (3.0, 4.5, "Un momento..."),
    (4.5, 6.6, "No, non tu. Quello che sta parlando."),
]


def turned(vector: np.ndarray, angle: float) -> np.ndarray:
    a = math.radians(angle)
    return np.array([vector[0] * math.cos(a) + vector[1] * math.sin(a), -vector[0] * math.sin(a) + vector[1] * math.cos(a)])


@dataclass(frozen=True)
class Moment:
    face: np.ndarray
    size: float
    scale: float
    presence: float
    reading: float
    turned_away: float
    lean: float
    hood: np.ndarray
    gaze: np.ndarray


@dataclass(frozen=True)
class Prepared:
    key: tuple[float, float, int]
    color: np.ndarray
    alpha: np.ndarray
    shift: np.ndarray


def _prepare(sprite: np.ndarray, key: tuple[float, float, int]) -> Prepared:
    scale, angle, _ = key
    warped = rotated(sprite, angle, scale)
    size = warped.shape[0]
    ys, xs = np.nonzero(warped[..., 3] > 2)
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    shift = np.array([(x0 + x1) / 2 - size / 2, (y0 + y1) / 2 - size / 2])
    part = warped[y0:y1, x0:x1]
    return Prepared(key, part[..., :3].copy(), cv2.cvtColor(part[..., 3], cv2.COLOR_GRAY2BGR), shift)


class ReaperEffect(OneShot):
    DURATION = 7.0

    def setup(self) -> None:
        self.cold = ColorGrade.vignette(self.w, self.h, COLD, brightness=0.82, depth=0.8, start=0.25, power=1.4)
        self.cloak, self.hood_offset = cloak(self.rng)
        self.skull = skull_in_shadow()
        self.scythe, self.grip = scythe()
        self.hand, self.pointing = bony_hand(False), bony_hand(True)
        self.prepared: dict[str, Prepared] = {}
        self.side = 1
        self.anchor: np.ndarray | None = None

    def begin(self, scene: Scene) -> None:
        self.anchor = None
        self.side = self._side_with_more_room(scene)

    def _side_with_more_room(self, scene: Scene) -> int:
        face_x = scene.face.center[0] if scene.face is not None else self.w * 0.4
        return 1 if face_x < self.w / 2 else -1

    def draw(self, frame: np.ndarray, scene: Scene, elapsed: float, dt: float) -> np.ndarray:
        power = envelope(elapsed, self.DURATION, 0.8, 0.5)
        out = self.cold.apply(desaturate(frame, 0.55 * power), power)
        self._track(scene)
        moment = self._moment(elapsed)
        out = self._behind_person(out, scene.person_mask, moment, elapsed)
        self._hold_props(out, moment, elapsed)
        self._subtitle(out, elapsed, power)
        return out

    def _track(self, scene: Scene) -> None:
        if scene.face is not None:
            face = scene.face
            target = np.array([face.center[0], face.center[1], face.height])
        else:
            target = np.array([self.w * 0.4, self.h * 0.42, self.h * 0.3])
        self.anchor = ease_toward(self.anchor, target, 0.15)

    def _moment(self, e: float) -> Moment:
        fx, fy, size = self.anchor
        side = self.side
        sunk = (1 - smoothstep(0.1, 1.2, e) + smoothstep(6.3, 7.0, e)) * 1.8 * size
        reading = smoothstep(3.0, 3.4, e) * (1 - smoothstep(4.5, 4.8, e))
        away = smoothstep(4.5, 5.0, e)
        return Moment(
            face=np.array([fx, fy]),
            size=size,
            scale=LARGER_THAN_YOU * size / SPRITE_FACE_HEIGHT,
            presence=smoothstep(0.1, 0.7, e) * (1 - smoothstep(6.4, 7.0, e)),
            reading=reading,
            turned_away=away,
            lean=side * (9 * smoothstep(0.8, 1.4, e) - 5 * reading - 22 * away),
            hood=np.array([fx + side * 1.55 * size, fy - 0.25 * size + sunk + 0.15 * size * reading]),
            gaze=np.array([-side * 0.07 * (1 - away) + side * 0.1 * away, 0.05 * reading]) * size,
        )

    def _behind_person(self, out: np.ndarray, person: np.ndarray, m: Moment, e: float) -> np.ndarray:
        side, size, scale = self.side, m.size, m.scale
        behind = out.copy()
        self._draw_scythe(behind, m.hood + np.array([-side * 0.85 * size, 1.35 * size]), 0.0, scale, m.presence)
        cloak_center = m.hood + turned(self.hood_offset * np.array([side, 1]) * scale, m.lean)
        self._blit_prepared(behind, "cloak", self._facing(self.cloak), cloak_center, scale, m.lean, m.presence)
        skull_at = m.hood + turned(m.gaze, m.lean)
        self._blit_prepared(behind, "skull", self.skull, skull_at, scale, m.lean, m.presence)
        self._draw_ember_eyes(behind, skull_at, m.lean, scale, m.presence, e)
        if m.turned_away > 0:
            self._point_at_someone_else(behind, m)
        return cv2.blendLinear(out, behind, person, 1 - person)

    def _point_at_someone_else(self, out: np.ndarray, m: Moment) -> None:
        side, size = self.side, m.size
        wrist = m.hood + np.array([side * (0.35 + 0.6 * m.turned_away) * size, 0.8 * size])
        opacity = m.presence * m.turned_away
        self._sleeve(out, m.hood + np.array([side * 0.3 * size, 1.25 * size]), wrist, 0.5 * size, opacity)
        outward = self.pointing if side < 0 else cv2.flip(self.pointing, 1)
        blit(out, outward, tuple(wrist + np.array([side * 0.32 * size, 0])), m.scale * 0.8, side * 6, opacity)

    def _hold_props(self, out: np.ndarray, m: Moment, e: float) -> None:
        side, size = self.side, m.size
        fx, fy = m.face
        shoulder = m.hood + np.array([-side * 0.5 * size, 1.2 * size])
        hourglass = envelope(e - 1.0, 2.1, 0.3, 0.3)
        if hourglass > 0:
            beside_you = np.array([fx + side * 0.85 * size, fy + 0.5 * size + 0.04 * size * math.sin(e * 5)])
            self._hold(out, emoji_sprite(HOURGLASS), shoulder, beside_you, 0.62 * size, m.presence * hourglass, e)
        scroll = envelope(e - 3.0, 1.6, 0.25, 0.3)
        if scroll > 0:
            under_the_hood = m.hood + np.array([-side * 0.35 * size, 1.0 * size])
            self._hold(out, emoji_sprite(SCROLL), shoulder, under_the_hood, 0.7 * size, m.presence * scroll, e)

    def _facing(self, sprite: np.ndarray) -> np.ndarray:
        return sprite if self.side > 0 else cv2.flip(sprite, 1)

    def _blit_prepared(
        self, out: np.ndarray, key: str, sprite: np.ndarray, center: np.ndarray, scale: float, angle: float, opacity: float
    ) -> None:
        wanted = (round(scale, 2), round(angle, 1), self.side)
        prepared = self.prepared.get(key)
        if prepared is None or prepared.key != wanted:
            prepared = self.prepared[key] = _prepare(sprite, wanted)
        if opacity <= 0:
            return
        color, alpha = prepared.color, prepared.alpha
        if opacity < 1:
            color, alpha = cv2.multiply(color, opacity), cv2.multiply(alpha, opacity)
        h, w = color.shape[:2]
        x0, y0 = int(center[0] + prepared.shift[0]) - w // 2, int(center[1] + prepared.shift[1]) - h // 2
        ox0, oy0, ox1, oy1 = clipped_box(x0, y0, x0 + w, y0 + h, self.w, self.h)
        if ox1 <= ox0 or oy1 <= oy0:
            return
        visible = np.s_[oy0 - y0 : oy1 - y0, ox0 - x0 : ox1 - x0]
        over(out[oy0:oy1, ox0:ox1], color[visible], alpha[visible])

    def _draw_scythe(self, out: np.ndarray, grip: np.ndarray, angle: float, scale: float, opacity: float) -> None:
        sprite = self._facing(self.scythe)
        h, w = sprite.shape[:2]
        pivot = np.array([self.grip[0] if self.side > 0 else w - self.grip[0], self.grip[1]])
        center = grip + turned((np.array([w, h]) / 2 - pivot) * scale, angle)
        self._blit_prepared(out, "scythe", sprite, center, scale, angle, opacity)
        blit(out, self._facing(self.hand), tuple(grip), scale * 0.75, angle, opacity)

    def _draw_ember_eyes(
        self, out: np.ndarray, skull_at: np.ndarray, lean: float, scale: float, opacity: float, e: float
    ) -> None:
        glow = opacity * (0.75 + 0.25 * math.sin(e * 13) * math.sin(e * 7.3))
        radius = max(2, int(8 * scale))
        for dx in (-20, 20):
            x, y = (skull_at + turned(np.array([dx, 10]) * scale, lean)).astype(int)
            cv2.circle(out, (x, y), radius * 2, (0, 0, int(80 * glow)), -1, cv2.LINE_AA)
            cv2.circle(out, (x, y), radius, (30, 40, int(255 * glow)), -1, cv2.LINE_AA)

    def _hold(
        self, out: np.ndarray, item: np.ndarray, shoulder: np.ndarray, target: np.ndarray, size: float, opacity: float, e: float
    ) -> None:
        if opacity <= 0.02:
            return
        wrist = target + np.array([self.side * 0.3 * size, 0.1 * size])
        self._sleeve(out, shoulder, wrist, 0.45 * size, opacity)
        tilt = 6 * math.sin(e * 2.3)
        blit(out, item, tuple(target), size / item.shape[0], tilt, opacity)
        fingers = target + np.array([self.side * 0.15 * size, 0.05 * size])
        blit(out, self._facing(self.hand), tuple(fingers), size / SPRITE_FACE_HEIGHT * 1.1, tilt, opacity)

    def _sleeve(self, out: np.ndarray, shoulder: np.ndarray, wrist: np.ndarray, width: float, opacity: float) -> None:
        thickness = max(3, int(width))
        x0, y0 = np.maximum(np.minimum(shoulder, wrist) - thickness, 0).astype(int)
        x1, y1 = np.minimum(np.maximum(shoulder, wrist) + thickness, [self.w, self.h]).astype(int)
        if x1 <= x0 or y1 <= y0:
            return
        roi = out[y0:y1, x0:x1]
        layer = roi.copy()
        arm = (np.array([shoulder, wrist]) - [x0, y0]).astype(np.int32)
        cv2.polylines(layer, [arm], False, CLOTH, thickness, cv2.LINE_AA)
        cv2.polylines(layer, [arm - [0, thickness // 3]], False, CLOTH_HIGHLIGHT, max(1, thickness // 6), cv2.LINE_AA)
        cv2.addWeighted(layer, opacity, roi, 1 - opacity, 0, dst=roi)

    def _subtitle(self, out: np.ndarray, e: float, power: float) -> None:
        for start, end, text in SUBTITLES:
            if start <= e < end:
                opacity = power * envelope(e - start, end - start, 0.15, 0.2)
                sprite = text_sprite(
                    text.format(nome=self.name), "serif_bold", int(42 * self.unit), SUBTITLE, int(4 * self.unit), SUBTITLE_OUTLINE
                )
                caption(out, sprite, self.h - 70 * self.unit, opacity)
