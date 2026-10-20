import time
from dataclasses import dataclass

import cv2
import numpy as np

from ...domain.scene import Face, Scene
from ..base import Effect
from ..kit.color import ColorGrade, crossfade, desaturate, grayscale
from ..kit.compositing import add_glow, blit, radial_glow
from ..kit.geometry import box_around, polygon_mask
from ..kit.sprites import text_sprite
from ..kit.timing import ease_toward, smoothstep
from .props import SOIL, grass_strip, grave_candle, gravestone, ground_shadow, moonlit_flowers, scrolling_fog

MOONLIGHT = (1.0, 0.8, 0.64)
ZOMBIE_SKIN = (0.62, 1.0, 0.68, 0)
EYE_BAGS = (40, 22, 34)
BLOOD = (28, 18, 170)
FLAME = (200, 245, 255)
BOUQUET, WILTED = "\U0001f490", "\U0001f940"
NIGHTFALL = 0.5
RESURRECTION = 5.0
DEBRIS_PER_SECOND = 45
DEBRIS_GRAVITY = 1900


@dataclass
class Debris:
    x: float
    y: float
    vx: float
    vy: float
    size: float
    color: tuple[int, int, int]
    outline: np.ndarray


class GraveEffect(Effect):
    def setup(self) -> None:
        u = self.unit
        self.moonlight = ColorGrade.vignette(self.w, self.h, MOONLIGHT, brightness=0.68, depth=0.7, start=0.3, power=1.5)
        self.ground = self.h - int(62 * u)
        self.grass = grass_strip(self.rng, u, self.w)
        self.fog = scrolling_fog(self.rng, u, self.w)
        self.candle = grave_candle(u)
        self.flame_glow = radial_glow(int(30 * u), (70, 190, 255), 2.5)
        self.candle_glow = radial_glow(int(110 * u), (25, 30, 120), 2.5)
        self.stone_x = self.w / 2
        self.candle_x, self.flowers_x = self.w / 2 + 255 * u, self.w / 2 - 270 * u
        self.stone: np.ndarray | None = None
        self.engraved_for: tuple[str, str] | None = None
        self.shadow = ground_shadow(u)
        self.bouquet, self.wilted = moonlit_flowers(BOUQUET), moonlit_flowers(WILTED)
        self.night = 0.0
        self.buried_at: float | None = None
        self.risen_at: float | None = None
        self.empty_shot: np.ndarray | None = None
        self.backdrop: np.ndarray | None = None
        self.flicker = 1.0
        self.debris: list[Debris] = []
        self.thrown = 0.0

    def render(self, frame: np.ndarray, scene: Scene, on: bool, t: float, dt: float) -> np.ndarray:
        if self.stone is None:
            self._engrave_stone()
        if on:
            return self._bury(frame, scene, t, dt)
        if self.buried_at is not None:
            self._start_rising(t)
        if self.risen_at is None:
            return frame
        since = t - self.risen_at
        if since >= RESURRECTION:
            self._finish_rising()
            return frame
        return self._resurrect(frame, scene, since, dt)

    def _bury(self, frame: np.ndarray, scene: Scene, t: float, dt: float) -> np.ndarray:
        if self.buried_at is None:
            self.buried_at, self.risen_at, self.empty_shot = t, None, None
            self._engrave_stone()
        if scene.face is None and not scene.hands:
            self.empty_shot = frame.copy()
        self.night = min(1.0, self.night + dt / NIGHTFALL)
        return self._graveyard(frame, t, t - self.buried_at)

    def _start_rising(self, t: float) -> None:
        self.buried_at, self.risen_at, self.backdrop = None, t, None
        self.debris, self.thrown = [], 0.0

    def _finish_rising(self) -> None:
        self.risen_at, self.night, self.empty_shot, self.backdrop = None, 0.0, None, None

    def _engrave_stone(self) -> None:
        epitaph = (self.name, time.strftime("%d.%m.%Y"))
        if epitaph != self.engraved_for:
            self.stone, self.engraved_for = gravestone(self.rng, self.unit, *epitaph), epitaph

    def _graveyard(self, frame: np.ndarray, t: float, buried_for: float) -> np.ndarray:
        out = self._by_moonlight(frame, self.night)
        self._draw_fog(out, t, self.night)
        still_underground = (1 - smoothstep(0, NIGHTFALL, buried_for)) * 0.3
        self._draw_cemetery(out, self.night, still_underground, 0.0)
        return out

    def _by_moonlight(self, frame: np.ndarray, night: float) -> np.ndarray:
        if night <= 0:
            return frame.copy()
        return self.moonlight.apply(desaturate(frame, 0.7 * night), night)

    def _draw_fog(self, out: np.ndarray, t: float, night: float) -> None:
        drift = int(t * 22 * self.unit) % self.w
        band = out[self.h - self.fog.shape[0] :]
        cv2.addWeighted(band, 1.0, self.fog[:, drift : drift + self.w], 0.55 * night, 0, dst=band)

    def _draw_cemetery(self, out: np.ndarray, night: float, sunk: float, jitter: float) -> None:
        self._draw_stone(out, night, sunk, jitter)
        self._draw_candle(out, night)
        self._draw_flowers(out, night)
        blit(out, self.grass, (self.w / 2, self.h - self.grass.shape[0] / 2), opacity=night)

    def _draw_stone(self, out: np.ndarray, night: float, sunk: float, jitter: float) -> None:
        u = self.unit
        blit(out, self.shadow, (self.stone_x, self.ground + 4 * u), opacity=0.8 * night * (1 - sunk))
        if self.stone is None or sunk >= 1:
            return
        height = self.stone.shape[0]
        dx, dy = self.rng.normal(0, jitter, 2) if jitter > 0 else (0.0, 0.0)
        top = self.ground - height + sunk * height + dy
        above_ground = int(min(height, self.ground + 6 * u - top))
        if above_ground > 4:
            blit(out, self.stone[:above_ground], (self.stone_x + dx, top + above_ground / 2), opacity=night)

    def _draw_candle(self, out: np.ndarray, night: float) -> None:
        u = self.unit
        self.flicker = ease_toward(self.flicker, self.rng.uniform(0.55, 1.0), 0.4)
        x, y = self.candle_x, self.ground - self.candle.shape[0] / 2 + 6 * u
        add_glow(out, self.candle_glow, (x, y), night * self.flicker)
        blit(out, self.candle, (x, y), opacity=night)
        flame = (x, y - self.candle.shape[0] / 2 - 7 * u)
        add_glow(out, self.flame_glow, flame, night * self.flicker)
        size = (5 + 1.5 * self.flicker) * u
        color = tuple(int(c * night) for c in FLAME)
        cv2.ellipse(out, (int(flame[0]), int(flame[1])), (int(size * 0.55), int(size * 1.6)), 0, 0, 360, color, -1, cv2.LINE_AA)

    def _draw_flowers(self, out: np.ndarray, night: float) -> None:
        u = self.unit
        blit(out, self.bouquet, (self.flowers_x, self.ground - 38 * u), 95 * u / self.bouquet.shape[0], 12, opacity=night)
        fallen = (self.candle_x + 70 * u, self.ground - 8 * u)
        blit(out, self.wilted, fallen, 62 * u / self.wilted.shape[0], -70, opacity=night)

    def _resurrect(self, frame: np.ndarray, scene: Scene, since: float, dt: float) -> np.ndarray:
        u = self.unit
        night = self.night * (1 - smoothstep(2.2, 3.4, since))
        undead = 1 - smoothstep(3.9, 4.9, since)
        zombie = self._zombie(frame, scene.face, undead)
        out = self._climb_out(frame, zombie, scene.person_mask, since)
        out = self._by_moonlight(out, night)
        self._draw_fog(out, since, night)
        self._update_debris(since, dt)
        self._draw_cemetery(out, night, smoothstep(0.15, 1.4, since) * 1.05, 6 * u * (1 - smoothstep(0.9, 1.4, since)))
        for d in self.debris:
            cv2.fillPoly(out, [(np.array([d.x, d.y]) + d.outline * d.size).astype(np.int32)], d.color, cv2.LINE_AA)
        self._titles(out, since)
        return out

    def _climb_out(self, frame: np.ndarray, zombie: np.ndarray, mask: np.ndarray, since: float) -> np.ndarray:
        underground = 1 - smoothstep(0.35, 1.8, since)
        if underground > 0:
            if self.backdrop is None:
                self.backdrop = self.empty_shot if self.empty_shot is not None else self._without_person(frame, mask)
            depth = int(underground * self.h * 0.92)
            body, silhouette = np.zeros_like(zombie), np.zeros_like(mask)
            body[depth:], silhouette[depth:] = zombie[: self.h - depth], mask[: self.h - depth]
            behind = self.backdrop
        else:
            body, silhouette = zombie, mask
            behind = crossfade(self.backdrop, frame, smoothstep(1.8, 2.3, since))
        return cv2.blendLinear(body, behind, silhouette, 1 - silhouette)

    def _zombie(self, frame: np.ndarray, face: Face | None, undead: float) -> np.ndarray:
        if undead <= 0:
            return frame
        out = cv2.addWeighted(frame, 1 - 0.7 * undead, grayscale(frame), 0.7 * undead, 12 * undead)
        out = cv2.multiply(out, tuple(1 - undead * (1 - c) for c in ZOMBIE_SKIN))
        if face is not None:
            self._darken_eye_bags(out, face, undead)
        return out

    def _darken_eye_bags(self, out: np.ndarray, face: Face, undead: float) -> None:
        for eye in face.eyes:
            center = eye.mean(axis=0)
            bag = center + (eye - center) * np.array([1.9, 3.2]) + [0, 0.1 * face.height * 0.25]
            box = box_around(bag, self.w, self.h, margin=10)
            x0, y0, x1, y1 = box
            if x1 - x0 < 4 or y1 - y0 < 4:
                continue
            shade = cv2.GaussianBlur(polygon_mask(bag, box), (0, 0), max(1.0, 0.12 * (x1 - x0)))[..., None] * 0.7 * undead
            roi = out[y0:y1, x0:x1]
            roi[:] = (roi * (1 - shade) + np.array(EYE_BAGS, np.float32) * shade).astype(np.uint8)

    def _without_person(self, frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
        small = cv2.resize(frame, (self.w // 8, self.h // 8), interpolation=cv2.INTER_AREA)
        hole = cv2.dilate((cv2.resize(mask, small.shape[1::-1]) > 0.3).astype(np.uint8), np.ones((5, 5), np.uint8))
        filled = cv2.inpaint(small, hole, 6, cv2.INPAINT_TELEA)
        return cv2.GaussianBlur(cv2.resize(filled, (self.w, self.h), interpolation=cv2.INTER_LINEAR), (0, 0), 6)

    def _update_debris(self, since: float, dt: float) -> None:
        if since < 1.5:
            self._throw_debris(since)
        for d in self.debris:
            d.vy += DEBRIS_GRAVITY * self.unit * dt
            d.x += d.vx * dt
            d.y += d.vy * dt
        self.debris = [d for d in self.debris if d.y < self.h + 30 * self.unit]

    def _throw_debris(self, since: float) -> None:
        u, rng = self.unit, self.rng
        for _ in range(int(since * DEBRIS_PER_SECOND) - int(self.thrown)):
            chip = rng.random() < 0.3 and since < 1.2
            self.debris.append(
                Debris(
                    x=self.stone_x + rng.normal(0, 190 if chip else 330) * u,
                    y=self.ground - rng.uniform(0, 160 if chip else 20) * u,
                    vx=rng.normal(0, 260) * u,
                    vy=-rng.uniform(450, 950) * u,
                    size=rng.uniform(5, 16) * u,
                    color=(int(rng.integers(95, 140)),) * 3 if chip else SOIL[int(rng.integers(len(SOIL)))],
                    outline=self._jagged_outline(),
                )
            )
        self.thrown = since * DEBRIS_PER_SECOND

    def _jagged_outline(self) -> np.ndarray:
        corners = int(self.rng.integers(5, 8))
        angles = np.sort(self.rng.uniform(0, 2 * np.pi, corners))
        radii = self.rng.uniform(0.6, 1.0, corners)
        return np.stack([np.cos(angles) * radii, np.sin(angles) * radii], axis=1)

    def _titles(self, out: np.ndarray, since: float) -> None:
        u = self.unit
        fade = 1 - smoothstep(4.2, 4.8, since)
        if since > 0.8:
            pop = smoothstep(0.8, 1.1, since)
            title = text_sprite("È RISORTO.", "gothic", int(118 * u), BLOOD, int(5 * u), (0, 0, 0))
            dx, dy = self.rng.normal(0, 2.5 * u, 2)
            blit(out, title, (self.w / 2 + dx, 120 * u + dy), 1.35 - 0.35 * pop, opacity=pop * fade)
        if since > 2.0:
            pop = smoothstep(2.0, 2.3, since)
            subtitle = text_sprite("Purtroppo.", "gothic", int(76 * u), (215, 230, 220), int(4 * u), (0, 0, 0))
            blit(out, subtitle, (self.w / 2, 242 * u), opacity=pop * fade)
