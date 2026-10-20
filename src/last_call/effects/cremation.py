import math

import cv2
import numpy as np

from ..domain.scene import Scene
from .base import OneShot
from .kit.color import ColorGrade, crossfade
from .kit.compositing import blit
from .kit.sprites import emoji_sprite, premultiply, text_sprite
from .kit.timing import envelope, smoothstep

URN = "⚱️"
EMBER = (40 / 255, 140 / 255, 1.0, 0)
SPARK = (40, 150, 255)
SMOKE_GRAY = 70
BRASS, BRASS_EDGE, ENGRAVING = (70, 165, 205), (30, 95, 140), (20, 45, 70)
STICKER, STICKER_TEXT = (40, 30, 215), (60, 235, 255)

CHAR, CHARRED, SNAPSHOT, CRUMBLED, EMBERS, URN_IN = 1.2, 2.8, 3.0, 5.0, 5.5, 5.6
ESTIMATE_BACKGROUND = SNAPSHOT - 0.2

FIRE_STOPS = [
    (0.0, (0, 0, 0)),
    (0.25, (0, 8, 110)),
    (0.45, (0, 55, 225)),
    (0.65, (10, 145, 255)),
    (0.85, (80, 225, 255)),
    (1.0, (200, 250, 255)),
]
FIRE_SCALE = 4
GRAIN = [(dx, dy) for dy in range(3) for dx in range(3)]


def periodic_noise(shape: tuple[int, int], scale: float, rng: np.random.Generator) -> np.ndarray:
    spectrum = np.fft.fft2(rng.standard_normal(shape))
    fy, fx = np.fft.fftfreq(shape[0])[:, None], np.fft.fftfreq(shape[1])[None, :]
    smooth = np.real(np.fft.ifft2(spectrum * np.exp(-((fx**2 + fy**2) * (2 * math.pi * scale) ** 2) / 2)))
    smooth = (smooth - smooth.mean()) / (smooth.std() + 1e-9)
    return np.clip(0.5 + smooth / 5, 0, 1).astype(np.float32)


def crack_map(shape: tuple[int, int], cells: int, rng: np.random.Generator) -> np.ndarray:
    h, w = shape
    seeds = np.ones(shape, np.uint8)
    seeds[rng.integers(0, h, cells), rng.integers(0, w, cells)] = 0
    _, labels = cv2.distanceTransformWithLabels(seeds, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
    edges = ((labels != np.roll(labels, 1, 0)) | (labels != np.roll(labels, 1, 1))).astype(np.float32)
    return cv2.GaussianBlur(_wobbled(edges, 0.25 * math.sqrt(h * w / cells), rng), (0, 0), 0.6)


def _wobbled(image: np.ndarray, amount: float, rng: np.random.Generator) -> np.ndarray:
    shape = image.shape
    yy, xx = np.mgrid[0 : shape[0], 0 : shape[1]].astype(np.float32)
    xx += (periodic_noise(shape, 6.0, rng) - 0.5) * 2 * amount
    yy += (periodic_noise(shape, 6.0, rng) - 0.5) * 2 * amount
    return cv2.remap(image, xx, yy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def _fire_palette() -> np.ndarray:
    heat = np.linspace(0, 1, 256)
    stops = [s for s, _ in FIRE_STOPS]
    return np.stack([np.interp(heat, stops, [c[i] for _, c in FIRE_STOPS]) for i in range(3)], axis=1).astype(np.uint8)


def _screen_blend(base: np.ndarray, light: np.ndarray) -> np.ndarray:
    return cv2.bitwise_not(cv2.multiply(cv2.bitwise_not(base), cv2.bitwise_not(light), scale=1 / 255))


def _ember_pulse(e: float) -> float:
    return 0.65 + 0.35 * math.sin(e * 9)


def _ember_light(glow: np.ndarray) -> np.ndarray:
    return cv2.multiply(cv2.cvtColor(cv2.convertScaleAbs(glow, alpha=255), cv2.COLOR_GRAY2BGR), EMBER)


def _offer_sticker(unit: float) -> np.ndarray:
    r = int(160 * unit)
    size = 2 * r + 8
    image = np.zeros((size, size, 4), np.uint8)
    angles = np.linspace(0, 2 * math.pi, 49)[:-1]
    radii = np.where(np.arange(48) % 2, 0.84 * r, r)
    star = np.stack([size / 2 + radii * np.cos(angles), size / 2 + radii * np.sin(angles)], axis=1).astype(np.int32)
    cv2.fillPoly(image, [star], (*STICKER, 255), cv2.LINE_AA)
    cv2.polylines(image, [star], True, (*STICKER_TEXT, 255), max(2, int(4 * unit)), cv2.LINE_AA)
    lines = [("OFFERTA DEL MESE", 20), ("3×2", 104), ("sulle urne", 34)]
    sprites = [text_sprite(text, "black", int(px * unit), STICKER_TEXT) for text, px in lines]
    y = size / 2 - sum(s.shape[0] for s in sprites) / 2 + 6 * unit
    for sprite in sprites:
        blit(image[..., :3], sprite, (size / 2, y + sprite.shape[0] / 2))
        y += sprite.shape[0] - 6 * unit
    return image


class Ashes:
    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.position = np.zeros((0, 2), np.float32)
        self.velocity = np.zeros((0, 2), np.float32)
        self.color = np.zeros((0, 3), np.uint8)
        self.age = np.zeros(0, np.float32)
        self.life = np.zeros(0, np.float32)

    def __len__(self) -> int:
        return len(self.position)

    @property
    def room(self) -> int:
        return self.limit - len(self)

    def add(self, position: np.ndarray, velocity: np.ndarray, color: np.ndarray, life: np.ndarray) -> None:
        self.position = np.vstack([self.position, position])
        self.velocity = np.vstack([self.velocity, velocity.astype(np.float32)])
        self.color = np.vstack([self.color, color])
        self.age = np.concatenate([self.age, np.zeros(len(position), np.float32)])
        self.life = np.concatenate([self.life, life.astype(np.float32)])

    def rise(self, dt: float, rng: np.random.Generator, unit: float, width: int, height: int) -> None:
        if not len(self):
            return
        self.age += dt
        self.velocity[:, 0] += rng.normal(0, 120 * unit * dt, len(self.velocity))
        self.velocity[:, 1] -= 60 * unit * dt
        self.velocity *= 1 - 0.6 * dt
        self.position += self.velocity * dt
        x, y = self.position[:, 0], self.position[:, 1]
        alive = (self.age < self.life) & (x >= 1) & (x < width - 3) & (y >= 1)
        alive &= y < height - 3
        self.position, self.velocity, self.color, self.age, self.life = (
            a[alive] for a in (self.position, self.velocity, self.color, self.age, self.life)
        )

    def draw(self, out: np.ndarray) -> None:
        if not len(self):
            return
        x, y = self.position[:, 0].astype(int), self.position[:, 1].astype(int)
        fade = (1 - self.age / self.life)[:, None]
        for dx, dy in GRAIN:
            under = out[y + dy, x + dx].astype(np.float32)
            out[y + dy, x + dx] = (under * (1 - fade) + self.color * fade).astype(np.uint8)


class CremationEffect(OneShot):
    DURATION = 8.0
    MAX_ASHES = 2500

    def setup(self) -> None:
        self.lw, self.lh = self.w // FIRE_SCALE, self.h // FIRE_SCALE
        self.flame_a = periodic_noise((self.lh, self.lw), 4.0, self.rng)
        self.flame_b = periodic_noise((self.lh, self.lw), 1.8, self.rng)
        self.smoke_noise = periodic_noise((self.lh, self.lw), 7.0, self.rng)
        self.row_position = (np.arange(self.lh, dtype=np.float32) / self.lh)[:, None]
        self.palette = _fire_palette()
        self.cracks = self._ember_cracks()
        self.ash_texture = self._ash_texture()
        self.firelight = ColorGrade(self._firelight())
        self.sticker = _offer_sticker(self.unit)
        self.urn = emoji_sprite(URN)

    def _ember_cracks(self) -> np.ndarray:
        half = (self.h // 2, self.w // 2)
        fine = crack_map(half, 420, self.rng) * self._here_and_there(half)
        plates = crack_map(half, 70, self.rng)
        return cv2.resize(np.clip(1.4 * np.maximum(plates, 0.8 * fine), 0, 1), (self.w, self.h))

    def _here_and_there(self, shape: tuple[int, int]) -> np.ndarray:
        return np.clip((periodic_noise(shape, 18.0, self.rng) - 0.45) * 4, 0, 1)

    def _ash_texture(self) -> np.ndarray:
        half = (self.h // 2, self.w // 2)
        patches = cv2.resize(periodic_noise(half, 5.0, self.rng), (self.w, self.h))
        return cv2.cvtColor((np.clip((patches - 0.45) * 2.2, 0, 1) * 75).astype(np.uint8), cv2.COLOR_GRAY2BGR)

    def _firelight(self) -> np.ndarray:
        y = np.linspace(0, 1, self.h, dtype=np.float32)[:, None, None]
        warm = np.array([0.5, 0.78, 1.0], np.float32)
        light = (0.35 + 0.35 * y) + warm * 0.45 * y**2
        return (np.clip(np.broadcast_to(light, (self.h, self.w, 3)), 0, 1) * 255).astype(np.uint8)

    def begin(self, scene: Scene) -> None:
        self.snapped = False
        self.plate: np.ndarray | None = None
        self.roi: tuple[slice, slice] | None = None
        self.urn_x = self.w / 2
        self.ashes = Ashes(self.MAX_ASHES)
        self.progress = 0.0
        self.name_plate = self._name_plate(self.name)

    def draw(self, frame: np.ndarray, scene: Scene, elapsed: float, dt: float) -> np.ndarray:
        e = elapsed
        if e >= ESTIMATE_BACKGROUND and self.plate is None:
            self.plate = self._empty_room(frame, scene.person_mask)
        base, embers = self._burn_person(frame, scene, e, dt)
        out = self._firelit(base, envelope(e, self.DURATION, 0.8, 1.0), e)
        if embers is not None:
            out = cv2.add(out, embers)
        self.ashes.rise(dt, self.rng, self.unit, self.w, self.h)
        self.ashes.draw(out)
        self._smoke(out, e, smoothstep(SNAPSHOT - 0.5, SNAPSHOT + 0.5, e) * (1 - smoothstep(6.0, self.DURATION, e)))
        if e >= URN_IN:
            self._urn(out, e)
        fire = smoothstep(0.0, 1.5, e) * (1 - smoothstep(EMBERS, EMBERS + 1.3, e))
        out = self._fire(out, e, fire)
        return crossfade(frame, out, envelope(e, self.DURATION, 0.2, 0.8))

    def _burn_person(self, frame: np.ndarray, scene: Scene, e: float, dt: float) -> tuple[np.ndarray, np.ndarray | None]:
        if e < SNAPSHOT:
            char = smoothstep(CHAR, CHARRED, e)
            if char <= 0:
                return frame, None
            return self._char(frame, scene.person_mask, char, e)
        if not self.snapped:
            self._snapshot(frame, scene.person_mask)
        return self._crumble(frame, scene.person_mask, e, dt)

    def _firelit(self, frame: np.ndarray, power: float, e: float) -> np.ndarray:
        k = power * (1 + 0.06 * math.sin(e * 17) * math.sin(e * 5.3))
        return self.firelight.apply(frame, min(1.0, k))

    def _to_ash(self, image: np.ndarray, roi: tuple[slice, slice] = (slice(None), slice(None))) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.add(cv2.cvtColor(cv2.convertScaleAbs(gray, alpha=0.3, beta=18), cv2.COLOR_GRAY2BGR), self.ash_texture[roi])

    def _char(self, frame: np.ndarray, mask: np.ndarray, char: float, e: float) -> tuple[np.ndarray, np.ndarray | None]:
        weight = mask * char
        out = cv2.blendLinear(self._to_ash(frame), frame, weight, 1 - weight)
        glow = smoothstep(CHAR + 0.5, CHARRED, e) * _ember_pulse(e)
        surely_person = mask * mask
        return out, _ember_light(self.cracks * (surely_person * glow)) if glow > 0 else None

    def _empty_room(self, frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
        small = cv2.resize(frame, (self.w // 8, self.h // 8), interpolation=cv2.INTER_AREA)
        hole = (cv2.resize(mask, (self.w // 8, self.h // 8)) > 0.2).astype(np.uint8) * 255
        hole = cv2.dilate(hole, np.ones((3, 3), np.uint8))
        filled = cv2.inpaint(small, hole, 3, cv2.INPAINT_TELEA)
        filled = cv2.GaussianBlur(cv2.resize(filled, (self.w // 4, self.h // 4)), (0, 0), 1.5)
        filled = cv2.resize(filled, (self.w, self.h), interpolation=cv2.INTER_LINEAR)
        soft = np.clip(cv2.resize(hole, (self.w, self.h)).astype(np.float32) / 255 * 1.3, 0, 1)
        return cv2.blendLinear(filled, frame, soft, 1 - soft)

    def _snapshot(self, frame: np.ndarray, mask: np.ndarray) -> None:
        self.snapped = True
        self.smoke_from = cv2.GaussianBlur(cv2.resize(mask, (self.lw, self.lh), interpolation=cv2.INTER_AREA), (0, 0), 4)
        rows, cols = np.nonzero(mask.max(axis=1) > 0.05)[0], np.nonzero(mask.max(axis=0) > 0.05)[0]
        if not len(rows):
            return
        self.roi = roi = (slice(rows[0], rows[-1] + 1), slice(cols[0], cols[-1] + 1))
        self.urn_x = float(np.clip((cols[0] + cols[-1]) / 2, 0.25 * self.w, 0.75 * self.w))
        part, statue_mask = frame[roi], mask[roi]
        self.statue = cv2.blendLinear(self._to_ash(part, roi), part, statue_mask, 1 - statue_mask)
        h, w = statue_mask.shape
        self.half = (max(1, w // 2), max(1, h // 2))
        self.statue_mask = cv2.resize(statue_mask, self.half, interpolation=cv2.INTER_AREA)
        self.statue_cracks = cv2.resize(self.cracks[roi], self.half, interpolation=cv2.INTER_AREA) * 1.5
        self.crumble_order = self._crumble_order()
        self.span = (float(self.crumble_order.min()), float(self.crumble_order.max()))
        self._place_ash_origins(rows[0], cols[0], h, w)

    def _crumble_order(self) -> np.ndarray:
        hh, hw = self.statue_mask.shape
        waves = cv2.resize(periodic_noise((max(8, hh // 4), max(8, hw // 4)), 2.0, self.rng), self.half)
        crumbs = cv2.GaussianBlur(self.rng.random((hh, hw), np.float32), (0, 0), 1.2)
        crumbs = (crumbs - crumbs.mean()) / (crumbs.std() + 1e-6)
        top_down = np.arange(hh, dtype=np.float32)[:, None] / hh
        return (top_down * 0.85 + 0.18 * waves + 0.035 * crumbs).astype(np.float32)

    def _place_ash_origins(self, top: int, left: int, h: int, w: int) -> None:
        hh, hw = self.statue_mask.shape
        gy, gx = np.mgrid[0:hh:2, 0:hw:2]
        inside = self.statue_mask[gy, gx] > 0.5
        gy, gx = gy[inside], gx[inside]
        by_release = np.argsort(self.crumble_order[gy, gx])
        gy, gx = gy[by_release], gx[by_release]
        self.ash_release = self.crumble_order[gy, gx]
        gy, gx = np.minimum(2 * gy, h - 1), np.minimum(2 * gx, w - 1)
        self.ash_origins = np.stack([gx + left, gy + top], axis=1).astype(np.float32)
        self.ash_colors = self.statue[gy, gx]

    def _crumble(self, frame: np.ndarray, live: np.ndarray, e: float, dt: float) -> tuple[np.ndarray, np.ndarray | None]:
        hidden = np.minimum(live * 1.6, 1.0)
        out = cv2.blendLinear(self.plate, frame, hidden, 1 - hidden)
        if self.roi is None:
            return out, None
        low, high = self.span
        progress = low + (high - low + 0.05) * smoothstep(SNAPSHOT, CRUMBLED, e) - 0.02
        embers = self._draw_statue(out, progress, e) if progress < high else None
        self._release_ashes(self.progress, progress, dt)
        self.progress = progress
        return out, embers

    def _draw_statue(self, out: np.ndarray, progress: float, e: float) -> np.ndarray:
        roi, size = self.roi, self.statue.shape[1::-1]
        standing = self.statue_mask * np.clip((self.crumble_order - progress) / 0.04, 0, 1)
        alive = cv2.resize(standing, size)
        out[roi] = cv2.blendLinear(self.statue, out[roi], alive, 1 - alive)
        crumbling_edge = np.clip(1 - np.abs(self.crumble_order - progress) / 0.025, 0, 1) * (0.7 * self.statue_mask)
        glow = self.statue_cracks * standing * _ember_pulse(e) + crumbling_edge
        embers = np.zeros_like(out)
        embers[roi] = _ember_light(cv2.resize(glow, size))
        return embers

    def _release_ashes(self, before: float, now: float, dt: float) -> None:
        if now <= before:
            return
        first, last = np.searchsorted(self.ash_release, [before, now])
        room = self.ashes.room
        if last <= first or room <= 0:
            return
        rng, u = self.rng, self.unit
        pick = rng.integers(first, last, min(room, last - first, int(9000 * dt) + 1))
        n = len(pick)
        velocity = np.stack([rng.normal(0, 40, n), -rng.uniform(20, 110, n)], axis=1) * u
        color = self.ash_colors[pick].copy()
        still_burning = rng.random(n) < 0.18
        color[still_burning] = np.array(SPARK, np.uint8)
        cold = ~still_burning
        color[cold] = np.clip(color[cold].astype(int) + rng.integers(10, 70, (int(cold.sum()), 1)), 0, 255)
        self.ashes.add(self.ash_origins[pick], velocity, color, rng.uniform(0.8, 2.2, n))

    def _scrolled(self, noise: np.ndarray, speed: float, e: float) -> np.ndarray:
        rows = np.arange(self.lh)
        return noise[(rows + int(e * speed * self.lh)) % self.lh]

    def _fire(self, out: np.ndarray, e: float, power: float) -> np.ndarray:
        if power <= 0.01:
            return out
        reach = 0.12 + 0.33 * power
        a = self._scrolled(self.flame_a, 0.9, e)
        b = self._scrolled(self.flame_b, 1.7, e)
        top = 1 - reach * (0.55 + 0.9 * self._tongues(e))
        base = np.clip((self.row_position - top) / (1 - top + 1e-3), 0, 1)
        heat = np.clip((base**0.8 * 1.45 - 0.55 * a - 0.45 * b) * power**0.5, 0, 1)
        fire = self.palette[(heat * 255).astype(np.uint8)]
        halo = cv2.GaussianBlur(fire, (0, 0), 5) // 2
        fire = cv2.resize(cv2.add(fire, halo), (self.w, self.h), interpolation=cv2.INTER_LINEAR)
        return _screen_blend(out, fire)

    def _tongues(self, e: float) -> np.ndarray:
        r = e * 12
        i, f = int(r), r - int(r)
        return ((1 - f) * self.flame_a[i % self.lh] + f * self.flame_a[(i + 1) % self.lh])[None, :]

    def _smoke(self, out: np.ndarray, e: float, power: float) -> None:
        if power <= 0.01:
            return
        n = self._scrolled(self.smoke_noise, 0.35, e)
        density = np.clip((n - 0.35) * 1.6, 0, 1) * (1.1 - self.row_position) * 0.55
        if self.snapped:
            density = density + 0.45 * self.smoke_from * np.clip((n - 0.2) * 2, 0, 1)
        density = np.clip(density * power, 0, 0.85)
        density = cv2.resize(density, (self.w, self.h))
        smoke = np.full_like(out, SMOKE_GRAY)
        out[:] = cv2.blendLinear(smoke, out, density, 1 - density)

    def _urn(self, out: np.ndarray, e: float) -> None:
        u = self.unit
        appear = smoothstep(URN_IN, URN_IN + 0.5, e)
        size = 300 * u
        x, y = self.urn_x, self.h - size / 2 - 22 * u
        bounce = 1 + 0.12 * math.sin(min(1.0, (e - URN_IN) / 0.5) * math.pi)
        blit(out, self.urn, (x, y), size / self.urn.shape[0] * bounce, opacity=appear)
        blit(out, self.name_plate, (x, y + 0.22 * size), opacity=appear)
        pop = smoothstep(URN_IN + 0.5, URN_IN + 0.8, e)
        toward_center = 1 if x < self.w / 2 else -1
        if pop > 0:
            wobble = 1 + 0.03 * math.sin(e * 6)
            blit(out, self.sticker, (x + toward_center * 310 * u, y - 70 * u), (0.6 + 0.4 * pop) * wobble, 12)

    def _name_plate(self, name: str) -> np.ndarray:
        u = self.unit
        text = text_sprite(name.upper(), "serif_bold", int(26 * u), ENGRAVING)
        max_w = int(150 * u)
        if text.shape[1] > max_w:
            text = cv2.resize(text, (max_w, max(1, int(text.shape[0] * max_w / text.shape[1]))), interpolation=cv2.INTER_AREA)
        pw, ph = text.shape[1] + int(26 * u), text.shape[0] + int(14 * u)
        plate = np.zeros((ph, pw, 4), np.uint8)
        screw_inset = int(8 * u)
        cv2.rectangle(plate, (0, 0), (pw - 1, ph - 1), (*BRASS, 255), -1)
        cv2.rectangle(plate, (1, 1), (pw - 2, ph - 2), (*BRASS_EDGE, 255), max(1, int(2 * u)), cv2.LINE_AA)
        for cx in (screw_inset, pw - screw_inset - 1):
            cv2.circle(plate, (cx, ph // 2), max(2, int(3 * u)), (*BRASS_EDGE, 255), -1, cv2.LINE_AA)
        shine = np.linspace(1.15, 0.8, ph, dtype=np.float32)[:, None, None]
        plate[..., :3] = np.clip(plate[..., :3] * shine, 0, 255).astype(np.uint8)
        blit(plate[..., :3], text, (pw / 2, ph / 2))
        return premultiply(plate)
