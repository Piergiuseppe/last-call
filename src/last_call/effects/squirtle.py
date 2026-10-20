import math
from dataclasses import dataclass
from typing import NamedTuple

import cv2
import numpy as np

from ..domain.scene import Oval, Scene
from ..gestures.detectors import squirtle_oval
from .base import Effect
from .kit.timing import ease_toward

WATER = (235, 190, 110)
FOAM = (255, 245, 225)
WHITE = (255, 255, 255)
DROP_TINT = np.array([1.06, 1.0, 0.92], np.float32)

NEAR_DEPTH = 0.45
LENS_DEPTH = 0.16
OVAL_PULL = 0.5


def _smoothed(old: Oval, new: Oval) -> Oval:
    shortest_turn = (new.angle - old.angle + 90) % 180 - 90
    return Oval(
        ease_toward(old.center, new.center, OVAL_PULL),
        ease_toward(old.axes, new.axes, OVAL_PULL),
        (old.angle + OVAL_PULL * shortest_turn) % 180,
    )


class Spray:
    FIELDS = ("origin", "offset", "velocity", "depth", "approach_speed", "radius_at_hands", "previous")

    def __init__(self) -> None:
        self.origin = np.zeros((0, 2))
        self.offset = np.zeros((0, 2))
        self.velocity = np.zeros((0, 2))
        self.depth = np.zeros(0)
        self.approach_speed = np.zeros(0)
        self.radius_at_hands = np.zeros(0)
        self.previous = np.zeros((0, 2))

    def __len__(self) -> int:
        return len(self.depth)

    @property
    def screen(self) -> np.ndarray:
        return self.origin + self.offset / self.depth[:, None]

    @property
    def radius(self) -> np.ndarray:
        return self.radius_at_hands / self.depth

    def add(self, origin, offset, velocity, approach_speed, radius_at_hands) -> None:
        n = len(offset)
        self.origin = np.vstack([self.origin, np.repeat(origin[None], n, axis=0)])
        self.offset = np.vstack([self.offset, offset])
        self.velocity = np.vstack([self.velocity, velocity])
        self.depth = np.concatenate([self.depth, np.ones(n)])
        self.approach_speed = np.concatenate([self.approach_speed, approach_speed])
        self.radius_at_hands = np.concatenate([self.radius_at_hands, radius_at_hands])
        self.previous = np.vstack([self.previous, origin + offset])

    def advance(self, dt: float, gravity: float, bounds: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
        self.previous = self.screen
        self.velocity[:, 1] += gravity * dt
        self.offset += self.velocity * dt
        self.depth = np.maximum(self.depth - self.approach_speed * dt, LENS_DEPTH * 0.5)
        pos, radius = self.screen, self.radius
        w, h = bounds
        visible = (pos[:, 0] > -radius) & (pos[:, 0] < w + radius) & (pos[:, 1] > -radius) & (pos[:, 1] < h + radius)
        on_lens = self.depth <= LENS_DEPTH
        self._keep(visible & ~on_lens)
        return pos[visible & on_lens], radius[visible & on_lens]

    def _keep(self, mask: np.ndarray) -> None:
        for name in self.FIELDS:
            setattr(self, name, getattr(self, name)[mask])


@dataclass
class LensDrop:
    pos: np.ndarray
    radius: float
    life: float
    phase: float
    age: float = 0.0
    vy: float = 0.0


class LensMaps(NamedTuple):
    sample_x: np.ndarray
    sample_y: np.ndarray
    shade: np.ndarray
    alpha: np.ndarray
    highlight: np.ndarray


class SquirtleEffect(Effect):
    DROPLETS_PER_SECOND = 900
    JET_DROPS_PER_SECOND = 90
    SQUIRTS_PER_SECOND = 2.5
    MAX_PARTICLES = 2000
    MAX_LENS_DROPS = 14

    def setup(self) -> None:
        self.spray = Spray()
        self.lens_drops: list[LensDrop] = []
        self.tracked: Oval | None = None
        self.oval: Oval | None = None
        self.power = 0.0
        self.lens_maps: dict[int, LensMaps] = {}

    @property
    def idle(self) -> bool:
        return self.power <= 0 and len(self.spray) == 0 and len(self.lens_drops) == 0

    def render(self, frame: np.ndarray, scene: Scene, on: bool, t: float, dt: float) -> np.ndarray:
        oval = self._follow_hands(scene, on)
        if oval is not None:
            self.oval = oval
            self.power = min(1.0, self.power + dt / 0.12)
        else:
            self.power = max(0.0, self.power - dt / 0.35)
        if self.idle:
            return frame

        pulse = 0.1 + 0.9 * (0.5 + 0.5 * math.sin(2 * math.pi * self.SQUIRTS_PER_SECOND * t)) ** 4
        if self.power > 0:
            self._emit(pulse, dt)
        landed = self.spray.advance(dt, gravity=450 * self.unit, bounds=(self.w, self.h))
        self._stick_to_lens(*landed)
        self._slide_lens_drops(dt)

        out = frame.copy()
        if self.power > 0:
            self._draw_jet(out, pulse)
        far = self.spray.depth > NEAR_DEPTH
        self._draw_far_droplets(out, far)
        out = self._draw_near_droplets(out, ~far)
        self._draw_lens_drops(out)
        return out

    def _follow_hands(self, scene: Scene, on: bool) -> Oval | None:
        detected = squirtle_oval(scene.hands)
        if detected is not None:
            self.tracked = detected if self.tracked is None else _smoothed(self.tracked, detected)
        elif not on:
            self.tracked = None
        if not on:
            return None
        return self.tracked if self.tracked is not None else self._oval_without_hands()

    def _oval_without_hands(self) -> Oval:
        return Oval(np.array([self.w / 2, self.h * 0.6]), np.array([70.0, 45.0]), 0.0)

    def _inside_oval(self, count: int, spread: float) -> np.ndarray:
        r = np.sqrt(self.rng.random(count)) * spread
        theta = self.rng.random(count) * 2 * math.pi
        a, b = self.oval.axes
        local = np.stack([np.cos(theta) * r * a, np.sin(theta) * r * b], axis=1)
        c, s = math.cos(math.radians(self.oval.angle)), math.sin(math.radians(self.oval.angle))
        return local @ np.array([[c, s], [-s, c]])

    def _emit(self, pulse: float, dt: float) -> None:
        if len(self.spray) >= self.MAX_PARTICLES:
            return
        scale = float(self.oval.axes.mean())
        self._emit_droplets(pulse, dt, scale)
        self._emit_jet(pulse, dt, scale)

    def _emit_droplets(self, pulse: float, dt: float, scale: float) -> None:
        rng = self.rng
        n = rng.poisson(self.DROPLETS_PER_SECOND * self.power * pulse * dt)
        offset = self._inside_oval(n, 0.85)
        outward = offset / (np.linalg.norm(offset, axis=1, keepdims=True) + 1e-6)
        velocity = outward * rng.uniform(0.8, 3.0, (n, 1)) * scale + rng.normal(0, 0.5 * scale, (n, 2))
        falls_like_a_fountain = rng.random(n) < 0.5
        slow, fast = rng.uniform(0.3, 0.8, n), rng.uniform(1.3, 2.2, n)
        approach = np.where(falls_like_a_fountain, slow, fast)
        self.spray.add(self.oval.center, offset, velocity, approach, rng.uniform(0.04, 0.09, n) * scale)

    def _emit_jet(self, pulse: float, dt: float, scale: float) -> None:
        rng = self.rng
        m = rng.poisson(self.JET_DROPS_PER_SECOND * self.power * pulse**2 * dt)
        offset = self._inside_oval(m, 0.3)
        velocity = rng.normal(0, 0.25 * scale, (m, 2))
        self.spray.add(self.oval.center, offset, velocity, rng.uniform(1.8, 2.6, m), rng.uniform(0.09, 0.15, m) * scale)

    def _stick_to_lens(self, positions: np.ndarray, radii: np.ndarray) -> None:
        u = self.unit
        for pos, r in zip(positions, radii):
            sticks = self.rng.random() <= np.clip(r / (45 * u), 0.03, 0.35)
            if not sticks:
                continue
            radius = float(np.clip(r * 1.4, 8 * u, 60 * u))
            if not self._merge_with_nearby_drop(pos, radius):
                self.lens_drops.append(LensDrop(pos.copy(), radius, self.rng.uniform(1.5, 3.5), self.rng.uniform(0, 6.3)))
        del self.lens_drops[: max(0, len(self.lens_drops) - self.MAX_LENS_DROPS)]

    def _merge_with_nearby_drop(self, pos: np.ndarray, radius: float) -> bool:
        for drop in self.lens_drops:
            if np.linalg.norm(drop.pos - pos) < max(drop.radius, radius) * 0.8:
                drop.radius = min(80 * self.unit, math.hypot(drop.radius, radius))
                return True
        return False

    def _slide_lens_drops(self, dt: float) -> None:
        u = self.unit
        for drop in self.lens_drops:
            drop.age += dt
            if drop.radius > 22 * u and drop.age > 0.4:
                drop.vy += 120 * u * (drop.radius / (40 * u)) * dt
                drop.pos[1] += drop.vy * dt
                drop.pos[0] += math.sin(drop.age * 4 + drop.phase) * 6 * u * dt
        self.lens_drops = [d for d in self.lens_drops if d.age < d.life and d.pos[1] - d.radius < self.h]

    def _draw_jet(self, out: np.ndarray, pulse: float) -> None:
        o = self.oval
        pad = int(o.axes.max() * 1.3)
        cx, cy = (int(v) for v in o.center)
        x0, y0 = max(0, cx - pad), max(0, cy - pad)
        x1, y1 = min(self.w, cx + pad), min(self.h, cy + pad)
        if x1 <= x0 or y1 <= y0:
            return
        roi = out[y0:y1, x0:x1]
        layer = roi.copy()
        center = (cx - x0, cy - y0)
        axes = tuple(int(v) for v in o.axes)
        cv2.ellipse(layer, center, axes, o.angle, 0, 360, WATER, -1, cv2.LINE_AA)
        inner = tuple(int(v * 0.55) for v in o.axes)
        cv2.ellipse(layer, center, inner, o.angle, 0, 360, FOAM, -1, cv2.LINE_AA)
        scale = float(o.axes.mean())
        for x, y in self._inside_oval(int(25 + 40 * pulse), 0.95) + center:
            r = max(1, int(self.rng.uniform(0.03, 0.12) * scale))
            cv2.circle(layer, (int(x), int(y)), r, WHITE, -1, cv2.LINE_AA)
        layer = cv2.GaussianBlur(layer, (0, 0), max(1.0, scale * 0.06))
        alpha = (0.4 + 0.35 * pulse) * self.power
        cv2.addWeighted(layer, alpha, roi, 1 - alpha, 0, dst=roi)

    def _draw_far_droplets(self, out: np.ndarray, mask: np.ndarray) -> None:
        if not mask.any():
            return
        layer = out.copy()
        for (x, y), (px, py), r in zip(self.spray.screen[mask], self.spray.previous[mask], self.spray.radius[mask]):
            head, tail = (int(x), int(y)), (int(px), int(py))
            cv2.line(layer, tail, head, WATER, max(1, int(r * 1.6)), cv2.LINE_AA)
            cv2.line(layer, tail, head, FOAM, max(1, int(r * 0.6)), cv2.LINE_AA)
            cv2.circle(layer, head, max(1, int(r * 0.8)), WHITE, -1, cv2.LINE_AA)
        cv2.addWeighted(layer, 0.85, out, 0.15, 0, dst=out)

    def _draw_near_droplets(self, out: np.ndarray, mask: np.ndarray) -> np.ndarray:
        if not mask.any():
            return out
        w2, h2 = self.w // 2, self.h // 2
        premultiplied = np.zeros((h2, w2, 3), np.uint8)
        alpha = np.zeros((h2, w2), np.uint8)
        spray = self.spray
        for (x, y), (px, py), r in zip(spray.screen[mask] / 2, spray.previous[mask] / 2, spray.radius[mask] / 2):
            head, tail, thick = (int(x), int(y)), (int(px), int(py)), max(1, int(r * 1.5))
            cv2.line(premultiplied, tail, head, tuple(c * 0.6 for c in FOAM), thick, cv2.LINE_AA)
            cv2.line(alpha, tail, head, 0.6 * 255, thick, cv2.LINE_AA)
            cv2.circle(premultiplied, head, max(1, int(r * 0.5)), (0.8 * 255,) * 3, -1, cv2.LINE_AA)
        out_of_focus = 2.5 * self.unit
        premultiplied = cv2.resize(cv2.GaussianBlur(premultiplied, (0, 0), out_of_focus), (self.w, self.h))
        alpha = cv2.resize(cv2.GaussianBlur(alpha, (0, 0), out_of_focus), (self.w, self.h))
        a = alpha[..., None].astype(np.float32) / 255
        return np.clip(out * (1 - a) + premultiplied, 0, 255).astype(np.uint8)

    def _lens(self, r: int) -> LensMaps:
        if r not in self.lens_maps:
            yy, xx = np.mgrid[-r : r + 1, -r : r + 1].astype(np.float32)
            rho = np.sqrt(xx**2 + yy**2) / r
            edge = np.clip(rho, 0, 1)
            magnification = 0.55 + 0.35 * edge**2
            shade = (1 - 0.2 * edge**6)[..., None]
            alpha = np.clip((1 - rho) / 0.3, 0, 1)[..., None]
            spot = 0.5 * np.exp(-(((xx + 0.3 * r) / (0.22 * r)) ** 2 + ((yy + 0.4 * r) / (0.15 * r)) ** 2))
            rim = 0.12 * np.exp(-(((rho - 0.8) / 0.1) ** 2)) * (yy > 0)
            self.lens_maps[r] = LensMaps(xx * magnification, yy * magnification, shade, alpha, ((spot + rim) * 255)[..., None])
        return self.lens_maps[r]

    def _draw_lens_drops(self, out: np.ndarray) -> None:
        if not self.lens_drops:
            return
        source = out.copy()
        for drop in self.lens_drops:
            r = int(drop.radius)
            lens = self._lens(r)
            cx, cy = (int(v) for v in drop.pos)
            x0, y0 = max(0, cx - r), max(0, cy - r)
            x1, y1 = min(self.w, cx + r + 1), min(self.h, cy + r + 1)
            if x1 <= x0 or y1 <= y0:
                continue
            sl = (slice(y0 - cy + r, y1 - cy + r), slice(x0 - cx + r, x1 - cx + r))
            seen = cv2.remap(
                source, cx + lens.sample_x[sl], cy + lens.sample_y[sl], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT
            )
            seen = cv2.GaussianBlur(seen, (0, 0), max(1.0, r * 0.12))
            color = seen * lens.shade[sl] * DROP_TINT + lens.highlight[sl]
            fade = min(1.0, (drop.life - drop.age) / 0.6)
            a = lens.alpha[sl] * fade
            roi = out[y0:y1, x0:x1]
            roi[:] = np.clip(roi * (1 - a) + color * a, 0, 255).astype(np.uint8)
