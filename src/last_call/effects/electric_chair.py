import math
from dataclasses import dataclass
from itertools import pairwise

import cv2
import numpy as np

from ..domain.landmarks import BodyPoint, HandPoint
from ..domain.scene import Body, Face, Hand, Scene
from .base import OneShot
from .kit.color import crossfade
from .kit.compositing import blit, shake
from .kit.geometry import box_around, degrees_of, normalized, pixel, polygon_mask
from .kit.sprites import emoji_sprite, premultiply, text_sprite
from .kit.timing import envelope, smoothstep

SKULL = "\U0001f480"
ARC_GLOW = (255, 160, 50)
ARC_CORE = (255, 245, 235)
SPARK = (170, 235, 255)
BONE = (235, 255, 245)
XRAY_TINT = (0.95, 0.85, 0.45)
STAMP_RED = (40, 30, 215)
SMOKE = (120, 120, 125)
STEEL = (205, 200, 192)
LEATHER = (38, 62, 98)
BRASS = (70, 170, 215)
OUTLINE = (20, 20, 24, 255)

SHOCKS = [(0.8, 0.55), (1.75, 0.6), (2.75, 0.7), (3.7, 0.5)]
XRAY_FLASHES = [(0.0, 0.2), (0.3, 0.4)]
CHARRED = 4.25
STAMP = 4.7
STAMP_TILT = 6
HEAD_INERTIA = 0.45

HELMET_DOME = 320
HELMET_RIM = (180, 232)
HELMET_SUPERSAMPLING = 2


def _span(first: HandPoint, last: HandPoint) -> list[HandPoint]:
    return [HandPoint(i) for i in range(first, last + 1)]


HAND_BONES = (
    [HandPoint.WRIST, *_span(HandPoint.THUMB_CMC, HandPoint.THUMB_TIP)],
    [HandPoint.WRIST, *_span(HandPoint.INDEX_MCP, HandPoint.INDEX_TIP)],
    _span(HandPoint.MIDDLE_MCP, HandPoint.MIDDLE_TIP),
    _span(HandPoint.RING_MCP, HandPoint.RING_TIP),
    [HandPoint.WRIST, *_span(HandPoint.PINKY_MCP, HandPoint.PINKY_TIP)],
    [HandPoint.INDEX_MCP, HandPoint.MIDDLE_MCP, HandPoint.RING_MCP, HandPoint.PINKY_MCP],
)
ARMS = (
    (BodyPoint.LEFT_SHOULDER, BodyPoint.LEFT_ELBOW, BodyPoint.LEFT_WRIST),
    (BodyPoint.RIGHT_SHOULDER, BodyPoint.RIGHT_ELBOW, BodyPoint.RIGHT_WRIST),
)


@dataclass
class Particle:
    x: float
    y: float
    vx: float
    vy: float
    age: float
    life: float
    size: float


@dataclass
class Head:
    rim: np.ndarray
    up: np.ndarray
    width: float
    height: float

    @classmethod
    def under_helmet(cls, face: Face) -> "Head":
        rim = face.forehead - face.up * 0.1 * face.height
        return cls(rim, face.up, face.width, face.height)

    @property
    def right(self) -> np.ndarray:
        return np.array([-self.up[1], self.up[0]])

    @property
    def angle(self) -> float:
        return _tilt_degrees(self.up)

    def at(self, across: float, above: float) -> np.ndarray:
        return self.rim + (self.right * across + self.up * above) * self.width

    def toward(self, new: "Head", inertia: float) -> "Head":
        return Head(
            inertia * self.rim + (1 - inertia) * new.rim,
            normalized(inertia * self.up + (1 - inertia) * new.up),
            inertia * self.width + (1 - inertia) * new.width,
            inertia * self.height + (1 - inertia) * new.height,
        )


@dataclass
class Cable:
    clamp: np.ndarray
    curve: np.ndarray


@dataclass
class Chest:
    neck: np.ndarray
    across: np.ndarray
    down: np.ndarray
    half_width: float

    @property
    def thickness(self) -> int:
        return max(3, int(0.07 * self.half_width))

    @property
    def angle(self) -> float:
        return degrees_of(self.across)

    def below_neck(self, k: float) -> np.ndarray:
        return self.neck + self.down * k * self.half_width


def _tilt_degrees(up: np.ndarray) -> float:
    return math.degrees(math.atan2(-up[0], -up[1]))


def _helmet() -> np.ndarray:
    s = HELMET_SUPERSAMPLING
    w, h = 360 * s, 250 * s
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32) / s
    color = np.zeros((h, w, 3), np.float32)
    alpha = np.zeros((h, w), np.float32)
    _paint(color, alpha, *_steel_dome(xx, yy), STEEL)
    _paint(color, alpha, *_leather_band(xx, yy), LEATHER)
    for region, shade in _brass_clamp(xx, yy):
        _paint(color, alpha, region, shade, BRASS)
    image = np.dstack([color, alpha * 255]).astype(np.uint8)
    _outline_helmet(image, s)
    return premultiply(cv2.resize(image, (360, 250), interpolation=cv2.INTER_AREA))


def _paint(color: np.ndarray, alpha: np.ndarray, region: np.ndarray, shade: np.ndarray, base: tuple) -> None:
    color[region] = (shade[region, None] * np.array(base)).clip(0, 255)
    alpha[region] = 1


def _steel_dome(xx: np.ndarray, yy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    nx, ny = (xx - 180) / 160, (200 - yy) / 140
    r2 = nx**2 + ny**2
    dome = (r2 <= 1) & (ny >= -0.02)
    nz = np.sqrt(np.clip(1 - r2, 0, 1))
    light = np.array([-0.45, 0.6, 0.66])
    lambert = np.clip(nx * light[0] + ny * light[1] + nz * light[2], 0, 1)
    return dome, 0.28 + 0.5 * lambert + 0.75 * lambert**24


def _leather_band(xx: np.ndarray, yy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    band = (yy >= 194) & (yy <= 232) & (np.abs(xx - 180) <= 172)
    return band, 0.55 + 0.45 * np.cos((xx - 180) / 172 * 1.3)


def _brass_clamp(xx: np.ndarray, yy: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    post = (np.abs(xx - 180) <= 16) & (yy >= 28) & (yy <= 70)
    nut = (np.abs(xx - 180) <= 28) & (yy >= 56) & (yy <= 74)
    return [(part, k * (0.45 + 0.55 * np.cos((xx - 180) / 30 * 1.4))) for part, k in ((post, 1.0), (nut, 0.8))]


def _outline_helmet(image: np.ndarray, s: int) -> None:
    cv2.ellipse(image, (180 * s, 200 * s), (160 * s, 140 * s), 0, 180, 360, OUTLINE, 3 * s, cv2.LINE_AA)
    cv2.rectangle(image, (8 * s, 194 * s), (352 * s, 232 * s), OUTLINE, 3 * s, cv2.LINE_AA)
    for x in (40, 110, 250, 320):
        cv2.circle(image, (x * s, 213 * s), 6 * s, (90, 180, 220, 255), -1, cv2.LINE_AA)
        cv2.circle(image, (x * s - 2 * s, 211 * s), 2 * s, (200, 240, 255, 255), -1, cv2.LINE_AA)
    cv2.rectangle(image, (152 * s, 56 * s), (208 * s, 74 * s), OUTLINE, 2 * s, cv2.LINE_AA)
    cv2.rectangle(image, (164 * s, 28 * s), (196 * s, 56 * s), OUTLINE, 2 * s, cv2.LINE_AA)


def _puff(rng: np.random.Generator) -> np.ndarray:
    size = 128
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    r = np.hypot(xx - size / 2, yy - size / 2) / (size / 2)
    noise = cv2.resize(rng.uniform(0.4, 1.0, (8, 8)).astype(np.float32), (size, size), interpolation=cv2.INTER_CUBIC)
    alpha = np.clip(np.exp(-(r**2) * 3) * noise, 0, 1)
    image = np.zeros((size, size, 4), np.uint8)
    image[..., :3] = SMOKE
    image[..., 3] = (alpha * 235).astype(np.uint8)
    return premultiply(image)


def _execution_stamp(unit: float, rng: np.random.Generator) -> np.ndarray:
    title = text_sprite("PENA ESEGUITA", "black", int(76 * unit), STAMP_RED)
    sub = text_sprite("venerdì, ore 17:30", "black", int(38 * unit), STAMP_RED)
    pad, gap, line = int(22 * unit), int(6 * unit), max(3, int(7 * unit))
    w = max(title.shape[1], sub.shape[1]) + 2 * pad
    h = title.shape[0] + sub.shape[0] + gap + 2 * pad
    image = np.zeros((h, w, 4), np.uint8)
    for sprite, y in ((title, pad), (sub, pad + title.shape[0] + gap)):
        x = (w - sprite.shape[1]) // 2
        image[y : y + sprite.shape[0], x : x + sprite.shape[1]] = sprite
    cv2.rectangle(image, (line, line), (w - line, h - line), (*STAMP_RED, 255), line, cv2.LINE_AA)
    inner = 3 * line
    cv2.rectangle(image, (inner, inner), (w - inner, h - inner), (*STAMP_RED, 255), max(1, line // 3), cv2.LINE_AA)
    return _tilted(_patchy_ink(image, rng), STAMP_TILT)


def _patchy_ink(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    h, w = image.shape[:2]
    speckle = (rng.random((h, w)) > 0.07).astype(np.float32)
    blotch = cv2.resize(rng.uniform(0.7, 1.0, (h // 24 + 2, w // 24 + 2)).astype(np.float32), (w, h))
    return (image.astype(np.float32) * (speckle * blotch)[..., None]).astype(np.uint8)


def _tilted(image: np.ndarray, angle: float) -> np.ndarray:
    h, w = image.shape[:2]
    rad = math.radians(angle)
    rw = int(w * abs(math.cos(rad)) + h * abs(math.sin(rad))) + 2
    rh = int(w * abs(math.sin(rad)) + h * abs(math.cos(rad))) + 2
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    matrix[:, 2] += (rw - w) / 2, (rh - h) / 2
    return cv2.warpAffine(image, matrix, (rw, rh), flags=cv2.INTER_LINEAR)


def _lightning(a: np.ndarray, b: np.ndarray, roughness: float, rng: np.random.Generator, depth: int = 5) -> np.ndarray:
    points = np.array([a, b], np.float32)
    spread = roughness
    for _ in range(depth):
        d = points[1:] - points[:-1]
        normal = np.stack([-d[:, 1], d[:, 0]], axis=1) / (np.linalg.norm(d, axis=1, keepdims=True) + 1e-6)
        mid = (points[1:] + points[:-1]) / 2 + normal * rng.normal(0, spread, (len(d), 1))
        merged = np.empty((len(points) + len(mid), 2), np.float32)
        merged[0::2], merged[1::2] = points, mid
        points, spread = merged, spread / 2
    return points


def _quadratic_curve(start: np.ndarray, bend: np.ndarray, end: np.ndarray, samples: int) -> np.ndarray:
    k = np.linspace(0, 1, samples)[:, None]
    return (1 - k) ** 2 * start + 2 * (1 - k) * k * bend + k**2 * end


def _bone(layer: np.ndarray, a: np.ndarray, b: np.ndarray, thick: int) -> None:
    cv2.line(layer, pixel(a), pixel(b), BONE, thick, cv2.LINE_AA)
    for p in (a, b):
        cv2.circle(layer, pixel(p), max(2, int(0.8 * thick)), BONE, -1, cv2.LINE_AA)


def _vertebra(layer: np.ndarray, center: np.ndarray, axes: tuple[int, int], angle: float) -> None:
    cv2.ellipse(layer, pixel(center), axes, angle, 0, 360, BONE, -1, cv2.LINE_AA)


class ElectricChairEffect(OneShot):
    DURATION = 6.5

    def setup(self) -> None:
        self.helmet = _helmet()
        self.glowing_helmet = np.clip(self.helmet * 1.35, 0, 255).astype(np.uint8)
        self.puff = _puff(self.rng)
        self.stamp = _execution_stamp(self.unit, self.rng)
        self.skull = emoji_sprite(SKULL)
        self.soot_noise = cv2.resize(self.rng.uniform(0.35, 1.0, (12, 12)).astype(np.float32), (128, 128))

    def begin(self, scene: Scene) -> None:
        self.head: Head | None = None
        self.sparks: list[Particle] = []
        self.smoke: list[Particle] = []

    @staticmethod
    def _shock(e: float) -> tuple[float, bool]:
        for start, length in SHOCKS:
            k = e - start
            if 0 <= k < length:
                return 1 - 0.4 * k / length, any(a <= k < b for a, b in XRAY_FLASHES)
        return 0.0, False

    def draw(self, frame: np.ndarray, scene: Scene, elapsed: float, dt: float) -> np.ndarray:
        e, u = elapsed, self.unit
        self._track(scene.face)
        shock, xray = self._shock(e)
        charred = smoothstep(CHARRED, CHARRED + 0.8, e)

        if xray:
            out = self._xray(frame)
            self._skeleton(out, scene)
        else:
            out = self._brownout(frame, e, shock)
            if charred > 0 and scene.face is not None:
                self._soot(out, scene.face, charred)
        cable = self._draw_helmet(out, e, glowing=xray) if self.head is not None else None
        if xray and scene.hands:
            self._draw_hand_bones(out, scene.hands)
        self._update_particles(e, dt, shock, charred)
        self._draw_sparks(out)
        if shock > 0 or (SHOCKS[0][0] < e < CHARRED and self.rng.random() < 0.12):
            self._arcs(out, cable, shock)
        self._draw_smoke(out)
        if e >= STAMP:
            self._draw_stamp(out, e)
        out = shake(out, 11 * u * shock + 8 * u * (STAMP <= e < STAMP + 0.18), self.rng)
        return crossfade(frame, out, envelope(e, self.DURATION, 0.25, 0.8))

    def _track(self, face: Face | None) -> None:
        if face is None:
            return
        seen = Head.under_helmet(face)
        self.head = seen if self.head is None else self.head.toward(seen, HEAD_INERTIA)

    def _brownout(self, frame: np.ndarray, e: float, shock: float) -> np.ndarray:
        if shock > 0:
            level = self.rng.uniform(0.3, 0.75)
        elif e < SHOCKS[0][0]:
            level = 1 - 0.25 * smoothstep(0.2, SHOCKS[0][0], e) * self.rng.uniform(0.5, 1.0)
        else:
            level = 0.82 + 0.12 * self.rng.random()
            if self.rng.random() < 0.08:
                level = self.rng.uniform(0.45, 0.7)
        return cv2.convertScaleAbs(frame, alpha=level)

    @staticmethod
    def _xray(frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        negative = cv2.convertScaleAbs(gray, alpha=-0.7, beta=175)
        return cv2.merge([cv2.convertScaleAbs(negative, alpha=k) for k in XRAY_TINT])

    def _skeleton(self, out: np.ndarray, scene: Scene) -> None:
        layer = np.zeros_like(out)
        self._torso_bones(layer, scene.body, scene.face)
        self._glow_bones(out, layer)
        if scene.face is not None:
            f = scene.face
            scale = 1.25 * f.width / self.skull.shape[1]
            blit(out, self.skull, f.center + f.up * 0.1 * f.height, scale, _tilt_degrees(f.up), 0.95)

    def _draw_hand_bones(self, out: np.ndarray, hands: list[Hand]) -> None:
        layer = np.zeros_like(out)
        for hand in hands:
            thick = max(2, int(0.07 * hand.size))
            for chain in HAND_BONES:
                for a, b in pairwise(chain):
                    _bone(layer, hand.points[a], hand.points[b], thick)
        self._glow_bones(out, layer)

    def _glow_bones(self, out: np.ndarray, layer: np.ndarray) -> None:
        glow = cv2.resize(layer, (self.w // 4, self.h // 4), interpolation=cv2.INTER_AREA)
        glow = cv2.GaussianBlur(glow, (0, 0), 3)
        glow = cv2.resize(glow, (self.w, self.h), interpolation=cv2.INTER_LINEAR)
        cv2.add(out, cv2.multiply(glow, (1.0, 0.9, 0.6, 0.0)), dst=out)
        cv2.max(out, layer, dst=out)

    @staticmethod
    def _shoulders(body: Body | None, face: Face | None) -> tuple[np.ndarray, np.ndarray] | None:
        if body is not None and body.shoulders_confidence > 0.5:
            return body.shoulders
        if face is None:
            return None
        base = face.chin - face.up * 0.75 * face.height
        across = np.array([-face.up[1], face.up[0]]) * 1.25 * face.width
        return base + across, base - across

    def _torso_bones(self, layer: np.ndarray, body: Body | None, face: Face | None) -> None:
        shoulders = self._shoulders(body, face)
        if shoulders is None:
            return
        left, right = shoulders
        middle = (left + right) / 2
        half = np.linalg.norm(left - right) / 2
        across = (left - right) / (2 * half + 1e-6)
        down = np.array([-across[1], across[0]])
        if down[1] < 0:
            down = -down
        chest = Chest(middle - down * 0.08 * half, across, down, half)
        if face is not None:
            self._neck(layer, face.chin, chest)
        self._collarbones(layer, chest, shoulders)
        self._spine(layer, chest)
        self._ribs(layer, chest)
        if body is not None:
            self._arms(layer, body, chest)

    @staticmethod
    def _neck(layer: np.ndarray, chin: np.ndarray, chest: Chest) -> None:
        half = chest.half_width
        for k in np.linspace(0.15, 0.9, 4):
            _vertebra(layer, chin + (chest.neck - chin) * k, (int(0.13 * half), int(0.045 * half)), chest.angle)

    @staticmethod
    def _collarbones(layer: np.ndarray, chest: Chest, shoulders: tuple[np.ndarray, np.ndarray]) -> None:
        neck, thick = chest.neck, chest.thickness
        for shoulder in shoulders:
            curve = np.array([neck, neck + (shoulder - neck) * 0.5 - chest.down * 0.06 * chest.half_width, shoulder])
            cv2.polylines(layer, [curve.astype(np.int32)], False, BONE, thick, cv2.LINE_AA)
            cv2.circle(layer, pixel(shoulder), int(1.4 * thick), BONE, -1, cv2.LINE_AA)

    @staticmethod
    def _spine(layer: np.ndarray, chest: Chest) -> None:
        half = chest.half_width
        for i in range(12):
            _vertebra(layer, chest.below_neck(0.18 + 0.2 * i), (int(0.11 * half), int(0.06 * half)), chest.angle)

    @staticmethod
    def _ribs(layer: np.ndarray, chest: Chest) -> None:
        half = chest.half_width
        for i in range(8):
            axes = (int(half * min(0.9, 0.5 + 0.1 * i)), int(half * (0.22 + 0.02 * i)))
            center = pixel(chest.below_neck(0.35 + 0.24 * i))
            cv2.ellipse(layer, center, axes, chest.angle, 190, 350, BONE, max(2, int(0.045 * half)), cv2.LINE_AA)

    @staticmethod
    def _arms(layer: np.ndarray, body: Body, chest: Chest) -> None:
        seen, thick = body.visibility, chest.thickness
        for shoulder, elbow, wrist in ARMS:
            if seen[shoulder] > 0.5 and seen[elbow] > 0.3:
                _bone(layer, body.points[shoulder], body.points[elbow], thick + 2)
                if seen[wrist] > 0.3:
                    radius_beside_ulna = chest.across * 0.035 * chest.half_width
                    for offset in (radius_beside_ulna, -radius_beside_ulna):
                        _bone(layer, body.points[elbow] + offset, body.points[wrist] + offset, thick)

    def _draw_helmet(self, out: np.ndarray, e: float, glowing: bool) -> Cable:
        head = self.head
        lowering = (1 - smoothstep(0.0, 0.45, e)) * 1.5 * head.height
        scale = 1.35 * head.width / HELMET_DOME
        rim = head.rim + head.up * lowering
        center = rim + head.up * (HELMET_RIM[1] - self.helmet.shape[0] / 2) * scale
        clamp = rim + head.up * (HELMET_RIM[1] - 28) * scale
        cable = Cable(clamp, self._draw_cable(out, clamp, head))
        blit(out, self.glowing_helmet if glowing else self.helmet, center, scale, head.angle)
        return cable

    def _draw_cable(self, out: np.ndarray, start: np.ndarray, head: Head) -> np.ndarray:
        u = self.unit
        out_of_frame = np.array([start[0] + 0.6 * head.width, -20 * u])
        sag = np.array([start[0] + 0.9 * head.width, start[1] - 0.45 * (start[1] - out_of_frame[1])])
        curve = _quadratic_curve(start, sag, out_of_frame, 24)
        cv2.polylines(out, [curve.astype(np.int32)], False, (18, 18, 18), max(3, int(9 * u)), cv2.LINE_AA)
        cv2.polylines(out, [(curve - 2 * u).astype(np.int32)], False, (85, 85, 85), max(1, int(2 * u)), cv2.LINE_AA)
        return curve

    def _arcs(self, out: np.ndarray, cable: Cable | None, shock: float) -> None:
        if cable is None:
            return
        paths = [p.astype(np.int32) for p in self._arc_paths(cable, shock)]
        everything = np.vstack(paths)
        margin = int(30 * self.unit)
        x0, y0, x1, y1 = box_around(everything, self.w, self.h, margin)
        if x1 - x0 < 4 or y1 - y0 < 4:
            return
        roi = out[y0:y1, x0:x1]
        local = [p - (x0, y0) for p in paths]
        glow = np.zeros_like(roi)
        cv2.polylines(glow, local, False, ARC_GLOW, max(3, int(9 * self.unit)), cv2.LINE_AA)
        glow = cv2.GaussianBlur(glow, (0, 0), 6 * self.unit)
        cv2.add(roi, glow, dst=roi)
        cv2.add(roi, glow, dst=roi)
        cv2.polylines(roi, local, False, ARC_CORE, max(1, int(2.5 * self.unit)), cv2.LINE_AA)

    def _arc_paths(self, cable: Cable, shock: float) -> list[np.ndarray]:
        u, head, rng = self.unit, self.head, self.rng
        paths = [_lightning(cable.curve[6], cable.clamp, 10 * u, rng, 4)]
        count = int(2 + 5 * shock) if shock > 0 else 1
        for _ in range(count):
            side = rng.choice([-1, 1])
            start = head.at(side * rng.uniform(0.3, 0.6), rng.uniform(0.05, 0.5))
            outward_and_down = rng.uniform(-0.6, 1.2)
            direction = head.right * side * math.cos(outward_and_down) - head.up * math.sin(outward_and_down)
            end = start + direction * rng.uniform(0.4, 1.1) * head.width
            paths.append(_lightning(start, end, 0.12 * head.width, rng))
        if shock > 0.5:
            paths.append(_lightning(cable.curve[3], head.at(rng.uniform(-0.5, 0.5), 0.1), 0.15 * head.width, rng))
        return paths

    def _draw_stamp(self, out: np.ndarray, e: float) -> None:
        slam = 1 - smoothstep(STAMP, STAMP + 0.12, e)
        blit(out, self.stamp, (self.w / 2, self.h * 0.8), 1 + 0.7 * slam, opacity=smoothstep(STAMP, STAMP + 0.06, e))

    def _update_particles(self, e: float, dt: float, shock: float, charred: float) -> None:
        if self.head is not None and shock > 0:
            self._emit_sparks(dt, shock)
        if self.head is not None and 0 < charred and e < self.DURATION - 1.0:
            self._emit_smoke(dt)
        u = self.unit
        for p in self.sparks:
            p.vy += 900 * u * dt
        for p in self.sparks + self.smoke:
            p.x += p.vx * dt
            p.y += p.vy * dt
            p.age += dt
        self.sparks = [p for p in self.sparks if p.age < p.life]
        self.smoke = [p for p in self.smoke if p.age < p.life]

    def _emit_sparks(self, dt: float, shock: float) -> None:
        u, rng = self.unit, self.rng
        for _ in range(rng.poisson(60 * dt * shock)):
            side = rng.choice([-1, 1])
            p = self.head.at(side * rng.uniform(0.3, 0.6), rng.uniform(0.0, 0.4))
            speed = rng.uniform(250, 650) * u
            angle = rng.uniform(0, 2 * math.pi)
            vx, vy = speed * math.cos(angle), speed * math.sin(angle) - 200 * u
            self.sparks.append(Particle(p[0], p[1], vx, vy, 0, rng.uniform(0.25, 0.55), 1))

    def _emit_smoke(self, dt: float) -> None:
        u, rng = self.unit, self.rng
        for _ in range(rng.poisson(16 * dt)):
            p = self.head.at(rng.uniform(-0.7, 0.7), rng.uniform(-0.4, 0.6))
            vx, vy = rng.normal(0, 20) * u, -rng.uniform(35, 80) * u
            self.smoke.append(Particle(p[0], p[1], vx, vy, 0, rng.uniform(1.4, 2.2), rng.uniform(60, 90) * u))

    def _draw_sparks(self, out: np.ndarray) -> None:
        for p in self.sparks:
            tail = (p.x - p.vx * 0.03, p.y - p.vy * 0.03)
            cv2.line(out, pixel((p.x, p.y)), pixel(tail), SPARK, max(1, int(2 * self.unit)), cv2.LINE_AA)

    def _draw_smoke(self, out: np.ndarray) -> None:
        for p in self.smoke:
            k = p.age / p.life
            size = p.size * (1 + 1.2 * k)
            sway = 12 * self.unit * math.sin(3 * p.age + p.size)
            blit(out, self.puff, (p.x + sway, p.y), size / self.puff.shape[0], opacity=(1 - k) * min(1, 4 * k + 0.3))

    def _soot(self, out: np.ndarray, face: Face, amount: float) -> None:
        oval = face.oval
        center = oval.mean(axis=0)
        outline = center + (oval - center) * 1.12 + face.up * 0.06 * face.height
        box = box_around(outline, self.w, self.h, margin=10)
        x0, y0, x1, y1 = box
        if x1 - x0 < 8 or y1 - y0 < 8:
            return
        mask = cv2.GaussianBlur(polygon_mask(outline, box), (0, 0), 0.08 * (x1 - x0))
        mask *= cv2.resize(self.soot_noise, (x1 - x0, y1 - y0))
        darkening = 1 - 0.7 * amount * mask
        roi = out[y0:y1, x0:x1]
        roi[:] = (roi * darkening[..., None]).astype(np.uint8)
