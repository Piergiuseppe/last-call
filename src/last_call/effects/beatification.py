import math
from dataclasses import dataclass
from typing import NamedTuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ..domain.scene import Face, Scene
from .base import OneShot
from .kit.compositing import blit, over
from .kit.geometry import normalized
from .kit.sprites import FONTS, emoji_sprite, row
from .kit.timing import envelope, smoothstep

GOLD = np.array([40, 195, 255], np.float32)
GOLD_LIGHT = np.array([185, 245, 255], np.float32)
GOLD_DEEP = np.array([10, 110, 195], np.float32)
OUTLINE = np.array([10, 30, 60], np.float32)
RAY_COLOR = np.array([120, 215, 255], np.float32) / 255
TITLE = "Dio"
DOG, PIG, SPARKLE, LILY = "\U0001f436", "\U0001f437", "✨", "⚜️"
DIDOT_BOLD = 2


def _gold_gradient(height: int) -> np.ndarray:
    k = np.linspace(0, 1, height, dtype=np.float32)[:, None, None]
    upper = GOLD_LIGHT + (GOLD - GOLD_LIGHT) * np.clip(k / 0.45, 0, 1)
    lower = GOLD + (GOLD_DEEP - GOLD) * np.clip((k - 0.45) / 0.55, 0, 1)
    return np.where(k < 0.45, upper, lower)


def _gold_text(text: str, size: int) -> np.ndarray:
    font = ImageFont.truetype(FONTS["didot"], size, index=DIDOT_BOLD)
    stroke = max(2, size // 22)
    left, top, right, bottom = font.getbbox(text, stroke_width=stroke)
    pad = stroke + 4
    shape = (right - left + 2 * pad, bottom - top + 2 * pad)
    origin = (pad - left, pad - top)
    outline, fill = Image.new("L", shape), Image.new("L", shape)
    ImageDraw.Draw(outline).text(origin, text, font=font, fill=255, stroke_width=stroke, stroke_fill=255)
    ImageDraw.Draw(fill).text(origin, text, font=font, fill=255)
    outline_alpha = np.array(outline, np.float32)[..., None] / 255
    fill_alpha = np.array(fill, np.float32)[..., None] / 255
    color = _gold_gradient(shape[1]) * fill_alpha + OUTLINE * (outline_alpha - fill_alpha)
    return np.dstack([color, outline_alpha * 255]).clip(0, 255).astype(np.uint8)


def _with_golden_glow(sprite: np.ndarray, radius: float, strength: float) -> np.ndarray:
    pad = int(radius * 2)
    canvas = cv2.copyMakeBorder(sprite, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(0, 0, 0, 0)).astype(np.float32)
    glow = np.clip(cv2.GaussianBlur(canvas[..., 3], (0, 0), radius) / 255 * strength, 0, 1)[..., None]
    alpha = canvas[..., 3:] / 255
    color = GOLD * glow * (1 - alpha) + canvas[..., :3]
    out_alpha = (alpha + glow * (1 - alpha)) * 255
    return np.dstack([color, out_alpha]).clip(0, 255).astype(np.uint8)


def _halo_sprite(width: int) -> np.ndarray:
    a, b = width // 2, max(4, width // 7)
    thick = max(3, width // 28)
    pad = thick * 4
    w, h = 2 * (a + pad), 2 * (b + pad)
    ring = np.zeros((h, w), np.float32)
    cv2.ellipse(ring, (w // 2, h // 2), (a, b), 0, 0, 360, 1.0, thick, cv2.LINE_AA)
    glow = cv2.GaussianBlur(ring, (0, 0), thick * 1.6) * 1.6
    alpha = np.clip(ring + glow, 0, 1)[..., None]
    color = np.minimum(ring[..., None] * GOLD_LIGHT + glow[..., None] * GOLD, alpha * 255)
    return np.dstack([color, alpha * 255]).astype(np.uint8)


def _blit_onto_layer(layer: np.ndarray, sprite: np.ndarray, center: tuple[float, float], scale: float) -> None:
    color = np.ascontiguousarray(layer[..., :3])
    alpha = np.ascontiguousarray(np.repeat(layer[..., 3:], 3, axis=2))
    blit(color, sprite, center, scale)
    solid = sprite.copy()
    solid[..., :3] = sprite[..., 3:]
    blit(alpha, solid, center, scale)
    layer[..., :3], layer[..., 3] = color, alpha[..., 0]


@dataclass
class Floater:
    x: float
    y: float
    vx: float
    vy: float
    size: float
    phase: float


class FrameStrip(NamedTuple):
    region: tuple[slice, slice]
    alpha: np.ndarray
    color: np.ndarray


class BeatificationEffect(OneShot):
    DURATION = 8.0
    PIGS = 9
    SPARKLES = 12
    RAYS_DOWNSCALE = 5

    def setup(self) -> None:
        u = self.unit
        self.rays_w, self.rays_h = self.w // self.RAYS_DOWNSCALE, self.h // self.RAYS_DOWNSCALE
        self.yy, self.xx = np.mgrid[0 : self.rays_h, 0 : self.rays_w].astype(np.float32)
        self.title = self._title_sprite()
        self.halo = _halo_sprite(int(300 * u))
        self.pig = emoji_sprite(PIG)
        self.sparkle = emoji_sprite(SPARKLE)
        self.frame_strips = self._holy_card_frame()

    def _title_sprite(self) -> np.ndarray:
        u = self.unit
        title = _gold_text(TITLE, int(120 * u))
        dog = emoji_sprite(DOG)
        dog_h = int(title.shape[0] * 0.8)
        dog = cv2.resize(dog, (int(dog.shape[1] * dog_h / dog.shape[0]), dog_h), interpolation=cv2.INTER_AREA)
        return _with_golden_glow(row(title, dog, gap=int(14 * u)), 12 * u, 0.9)

    def _holy_card_frame(self) -> list[FrameStrip]:
        u, w, h = self.unit, self.w, self.h
        border = np.zeros((h, w, 4), np.float32)
        lines = np.zeros((h, w), np.float32)
        outer, inner = int(14 * u), int(28 * u)
        cv2.rectangle(lines, (outer, outer), (w - outer, h - outer), 1.0, max(3, int(7 * u)), cv2.LINE_AA)
        cv2.rectangle(lines, (inner, inner), (w - inner, h - inner), 1.0, max(1, int(2 * u)), cv2.LINE_AA)
        glow = cv2.GaussianBlur(lines, (0, 0), 6 * u) * 0.9
        alpha = np.clip(lines + glow, 0, 1)
        shade = _gold_gradient(h) * np.ones((1, w, 1), np.float32)
        border[..., :3] = np.minimum(lines[..., None] * shade + glow[..., None] * GOLD, alpha[..., None] * 255)
        border[..., 3] = alpha * 255
        border = border.astype(np.uint8)
        self._add_corner_lilies(border)
        return self._edge_strips(border)

    def _add_corner_lilies(self, border: np.ndarray) -> None:
        u = self.unit
        lily = emoji_sprite(LILY)
        corner = 46 * u
        for x in (corner, self.w - corner):
            for y in (corner, self.h - corner):
                _blit_onto_layer(border, lily, (x, y), 52 * u / lily.shape[0])

    def _edge_strips(self, border: np.ndarray) -> list[FrameStrip]:
        s, w, h = int(72 * self.unit), self.w, self.h
        regions = (np.s_[:s, :], np.s_[h - s :, :], np.s_[s : h - s, :s], np.s_[s : h - s, w - s :])
        return [
            FrameStrip(r, np.ascontiguousarray(np.repeat(border[r][..., 3:], 3, axis=2)), border[r][..., :3].copy())
            for r in regions
        ]

    def begin(self, scene: Scene) -> None:
        rng, u, face = self.rng, self.unit, scene.face
        self.title_y = self._title_y(face)
        self.pigs = self._release_pigs(face)
        self.sparkles = [
            Floater(
                rng.uniform(0.04, 0.96) * self.w,
                rng.uniform(0.04, 0.96) * self.h,
                0,
                0,
                rng.uniform(30, 55) * u,
                rng.uniform(0, 6.3),
            )
            for _ in range(self.SPARKLES)
        ]

    def _title_y(self, face: Face | None) -> float:
        u, half = self.unit, self.title.shape[0] / 2
        top = 20 * u + half
        halo_under_title = face is not None and face.crown[1] - 0.3 * face.width < top + half
        return self.h - 20 * u - half if halo_under_title else top

    def _release_pigs(self, face: Face | None) -> list[Floater]:
        rng, u = self.rng, self.unit
        pigs: list[Floater] = []
        while len(pigs) < self.PIGS:
            x, y = rng.uniform(0.05, 0.95) * self.w, rng.uniform(0.1, 0.92) * self.h
            if face is not None and math.dist((x, y), face.center) < face.width * 1.3:
                continue
            vx = rng.choice([-1, 1]) * rng.uniform(15, 45) * u
            pigs.append(Floater(x, y, vx, rng.uniform(-25, 10) * u, rng.uniform(55, 95) * u, rng.uniform(0, 6.3)))
        return pigs

    def draw(self, frame: np.ndarray, scene: Scene, elapsed: float, dt: float) -> np.ndarray:
        fade = envelope(elapsed, self.DURATION, 0.6, 0.8)
        face = scene.face
        origin = face.crown if face is not None else np.array([self.w / 2, self.h * 0.15])
        out = cv2.add(frame, self._divine_light(scene, origin, elapsed, fade))
        self._draw_frame(out, fade)
        for pig in self.pigs:
            self._fly(pig, face, dt)
            self._draw_pig(out, pig, elapsed, fade)
        self._draw_sparkles(out, elapsed, fade)
        if face is not None:
            self._draw_head_halo(out, face, origin, elapsed, fade)
        self._draw_title(out, elapsed, fade)
        return out

    def _divine_light(self, scene: Scene, origin: np.ndarray, elapsed: float, fade: float) -> np.ndarray:
        q = self.RAYS_DOWNSCALE
        dx, dy = self.xx - origin[0] / q, self.yy - origin[1] / q
        theta = np.arctan2(dy, dx)
        r = np.sqrt(dx * dx + dy * dy)
        spin = 0.25 * elapsed
        rays = (0.5 + 0.5 * np.cos(12 * (theta + spin))) ** 6
        rays += 0.5 * (0.5 + 0.5 * np.cos(7 * (theta - 0.6 * spin) + 1.0)) ** 8
        rays *= np.exp(-r / (0.75 * self.rays_h))
        glare = np.exp(-((r / (0.12 * self.rays_w)) ** 2))
        behind_person = 1 - cv2.resize(scene.person_mask, (self.rays_w, self.rays_h), interpolation=cv2.INTER_AREA)
        intensity = 0.8 * rays * behind_person + 0.35 * glare + 0.05
        light = (intensity[..., None] * RAY_COLOR * 255 * fade).clip(0, 255).astype(np.uint8)
        return cv2.resize(light, (self.w, self.h), interpolation=cv2.INTER_LINEAR)

    def _draw_frame(self, out: np.ndarray, fade: float) -> None:
        for strip in self.frame_strips:
            alpha, color = strip.alpha, strip.color
            if fade < 0.999:
                alpha, color = cv2.convertScaleAbs(alpha, alpha=fade), cv2.convertScaleAbs(color, alpha=fade)
            over(out[strip.region], color, alpha)

    def _fly(self, pig: Floater, face: Face | None, dt: float) -> None:
        pig.x = (pig.x + pig.vx * dt + pig.size) % (self.w + 2 * pig.size) - pig.size
        pig.y = (pig.y + pig.vy * dt + pig.size) % (self.h + 2 * pig.size) - pig.size
        if face is not None:
            self._push_off_face(pig, face, dt)

    @staticmethod
    def _push_off_face(pig: Floater, face: Face, dt: float) -> None:
        away = np.array([pig.x, pig.y]) - face.center
        overlap = face.width * 1.1 + pig.size * 0.5 - np.linalg.norm(away)
        if overlap > 0:
            pig.x, pig.y = np.array([pig.x, pig.y]) + normalized(away) * overlap * min(1.0, 4 * dt)

    def _draw_pig(self, out: np.ndarray, pig: Floater, elapsed: float, fade: float) -> None:
        y = pig.y + 12 * self.unit * math.sin(1.6 * elapsed + pig.phase)
        angle = 12 * math.sin(1.1 * elapsed + pig.phase)
        blit(out, self.pig, (pig.x, y), pig.size / self.pig.shape[0], angle, fade)
        tilt = math.radians(angle)
        above_head = (pig.x - math.sin(tilt) * pig.size * 0.62, y - math.cos(tilt) * pig.size * 0.62)
        blit(out, self.halo, above_head, pig.size * 0.8 / self.halo.shape[1], angle, fade)

    def _draw_sparkles(self, out: np.ndarray, elapsed: float, fade: float) -> None:
        for s in self.sparkles:
            twinkle = max(0.0, math.sin(2.3 * elapsed + s.phase)) ** 2
            blit(out, self.sparkle, (s.x, s.y), s.size / self.sparkle.shape[0] * (0.6 + 0.4 * twinkle), 0, fade * twinkle)

    def _draw_head_halo(self, out: np.ndarray, face: Face, origin: np.ndarray, elapsed: float, fade: float) -> None:
        ux, uy = face.up
        tilt = -math.degrees(math.atan2(ux, -uy))
        bob = face.up * 5 * self.unit * math.sin(2.5 * elapsed)
        blit(out, self.halo, origin + bob, face.width * 1.25 / self.halo.shape[1], tilt, fade)

    def _draw_title(self, out: np.ndarray, elapsed: float, fade: float) -> None:
        pop = 0.85 + 0.15 * smoothstep(0, 0.5, elapsed)
        pulse = 1 + 0.03 * math.sin(3 * elapsed)
        y = self.title_y + 6 * self.unit * math.sin(2 * elapsed)
        blit(out, self.title, (self.w / 2, y), pop * pulse, opacity=fade)
