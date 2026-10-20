import math

import cv2
import numpy as np

from ..domain.scene import Scene
from .base import OneShot
from .kit.compositing import blit, shake
from .kit.sprites import emoji_sprite
from .kit.timing import smoothstep

DIVINE_HAND, MIDDLE_FINGER, DIZZY = "\U0001faf3", "\U0001f595", "\U0001f4ab"
GOLD = (90, 205, 255)
SKY_GREY = (225, 205, 195)
WARM_WHITE = (250, 252, 255)
SLAP = 1.3
FINGER_IN, FINGER_OUT = 2.4, 3.4
UPSIDE_DOWN = 180
AVERAGE_HEAD = 136


class HandOfGodEffect(OneShot):
    DURATION = 4.5

    def setup(self) -> None:
        self.hand = emoji_sprite(DIVINE_HAND)
        self.finger = emoji_sprite(MIDDLE_FINGER)
        self.star = emoji_sprite(DIZZY)
        self.clouds = self._cloud_bank()
        self.rays: np.ndarray | None = None

    def begin(self, scene: Scene) -> None:
        crown, _ = self._head(scene)
        self.rays = self._rays_from_above(crown[0])

    def draw(self, frame: np.ndarray, scene: Scene, elapsed: float, dt: float) -> np.ndarray:
        e = elapsed
        crown, head = self._head(scene)
        clouds_down = smoothstep(0, 0.6, e) * (1 - smoothstep(3.8, 4.5, e))

        out = frame.copy()
        glow = 0.45 + 0.35 * smoothstep(0.5, SLAP, e) * (1 - smoothstep(SLAP, 2.2, e)) + 0.05 * math.sin(9 * e)
        cv2.addWeighted(out, 1.0, self.rays, clouds_down * glow, 0, dst=out)
        height = self.clouds.shape[0]
        drift = 20 * self.unit * math.sin(0.7 * e)
        blit(out, self.clouds, (self.w / 2 + drift, height / 2 - (1 - clouds_down) * height), opacity=clouds_down)
        self._draw_slap(out, crown, head, e)
        self._draw_finger(out, crown, head, e)
        self._draw_stars(out, crown, head, e)

        if e >= SLAP:
            return self._impact(out, e - SLAP)
        return shake(out, 2 * self.unit * smoothstep(0.5, SLAP, e), self.rng)

    def _impact(self, out: np.ndarray, since: float) -> np.ndarray:
        flash = 0.85 * math.exp(-since / 0.08)
        out = cv2.add(out, (flash * 255,) * 3 + (0,))
        return shake(out, 24 * self.unit * math.exp(-since / 0.2), self.rng)

    def _head(self, scene: Scene) -> tuple[np.ndarray, float]:
        if scene.face is None:
            return np.array([self.w / 2, self.h * 0.3]), AVERAGE_HEAD * self.unit
        return scene.face.crown, scene.face.height

    def _draw_slap(self, out: np.ndarray, crown: np.ndarray, head: float, e: float) -> None:
        scale = 2.2 * head / self.hand.shape[1]
        height = self.hand.shape[0] * scale
        start = -height
        landing = max(crown[1] - 0.42 * height, 0.5 * height + 10 * self.unit)
        if e < SLAP:
            swing = min(1.0, max(0.0, (e - 0.5) / (SLAP - 0.5))) ** 1.7
            fingers_raised = 14 * (1 - swing)
        else:
            swing = 1 - smoothstep(SLAP + 0.15, 2.2, e)
            fingers_raised = -4 * swing
        if swing <= 0:
            return
        blit(out, self.hand, (crown[0], start + (landing - start) * swing), scale, fingers_raised)

    def _draw_finger(self, out: np.ndarray, crown: np.ndarray, head: float, e: float) -> None:
        shown = smoothstep(FINGER_IN, FINGER_IN + 0.3, e) * (1 - smoothstep(FINGER_OUT - 0.3, FINGER_OUT, e))
        if shown <= 0:
            return
        scale = 1.6 * head / self.finger.shape[0]
        height = self.finger.shape[0] * scale
        tip_on_forehead = max(crown[1] - 0.2 * height, 0.45 * height)
        y = -height / 2 + (tip_on_forehead + height / 2) * shown
        blit(out, self.finger, (crown[0], y), scale, UPSIDE_DOWN + 6 * math.sin(10 * e))

    def _draw_stars(self, out: np.ndarray, crown: np.ndarray, head: float, e: float) -> None:
        shown = smoothstep(SLAP, SLAP + 0.15, e) * (1 - smoothstep(SLAP + 1.3, SLAP + 1.6, e))
        if shown <= 0:
            return
        center = crown + np.array([0, 0.25 * head])
        for i in range(4):
            theta = 5 * e + i * math.pi / 2
            perspective = 0.8 + 0.2 * math.sin(theta)
            pos = center + np.array([0.75 * head * math.cos(theta), 0.18 * head * math.sin(theta)])
            scale = 0.45 * head / self.star.shape[0] * perspective
            blit(out, self.star, pos, scale, math.degrees(theta) % 360, shown)

    def _cloud_bank(self) -> np.ndarray:
        w, h = self.w + int(80 * self.unit), int(self.h * 0.3)
        alpha = self._cloud_cover(w, h)
        shift = int(0.12 * h)
        cover_above = np.zeros_like(alpha)
        cover_above[:-shift] = alpha[shift:]
        sunlit = cv2.GaussianBlur(cover_above, (0, 0), 0.04 * h)[..., None]
        color = (1 - sunlit) * np.array(SKY_GREY, np.float32) + sunlit * np.array(WARM_WHITE, np.float32)
        bgra = np.dstack([color * alpha[..., None], alpha * 255])
        return np.clip(bgra, 0, 255).astype(np.uint8)

    def _cloud_cover(self, w: int, h: int) -> np.ndarray:
        alpha = np.zeros((h, w), np.float32)
        alpha[: int(h * 0.3)] = 1
        for _ in range(70):
            x, y = self.rng.uniform(-0.05, 1.05) * w, self.rng.uniform(0.2, 0.62) * h
            cv2.circle(alpha, (int(x), int(y)), int(self.rng.uniform(0.1, 0.24) * h), 1.0, -1, cv2.LINE_AA)
        return cv2.GaussianBlur(alpha, (0, 0), 0.015 * h)

    def _rays_from_above(self, x: float) -> np.ndarray:
        scale = 4
        w, h = self.w // scale, self.h // scale
        layer = np.zeros((h, w, 3), np.float32)
        origin = np.array([x / scale, -0.15 * h])
        length = 1.6 * h
        for i, angle in enumerate(np.linspace(-38, 38, 11)):
            spread = math.radians(2.2 if i % 2 else 3.6)
            a = math.radians(angle + self.rng.uniform(-2, 2))
            left = origin + length * np.array([math.sin(a - spread), math.cos(a - spread)])
            right = origin + length * np.array([math.sin(a + spread), math.cos(a + spread)])
            strength = self.rng.uniform(0.5, 1.0)
            cv2.fillPoly(layer, [np.array([origin, left, right]).astype(np.int32)], [c * strength for c in GOLD])
        layer = cv2.GaussianBlur(layer, (0, 0), 2.5)
        stronger_at_top = np.clip(1 - np.linspace(0, 1, h) ** 0.8, 0, 1)[:, None, None]
        return cv2.resize(np.clip(layer * stronger_at_top, 0, 255).astype(np.uint8), (self.w, self.h))
