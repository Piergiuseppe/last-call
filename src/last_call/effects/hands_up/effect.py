import math

import cv2
import numpy as np

from ...domain.scene import Body, Face, Scene
from ..base import OneShot
from ..kit.color import crossfade, desaturate, radial_falloff
from ..kit.compositing import add_glow, blit, caption, radial_glow
from ..kit.geometry import clipped_box, normalized, pixel
from ..kit.sprites import text_sprite
from ..kit.timing import ease_toward, envelope, smoothstep
from .blood import BULLET_HOLE, SPLAT_RADIUS, SPLAT_STRETCHES, Wound, paint_entry_wound, splat_sprite, stamp_onto
from .scope import draw_reticle, look_through_scope
from .spatter import LensSpatter

RED = (40, 40, 255)
DOT_CORE = (215, 215, 255)
DOT_GLOW = (30, 30, 255)
BEAM = (0, 0, 110)
ALARM = (0, 0, 120)
WHITE, BLACK = (255, 255, 255), (0, 0, 0)
SCREEN_UP = (0.0, -1.0)

DOTS = 4
FOREHEAD_DOT, HEART_DOT = 0, 2
LOCK = (1.2, 2.3)
SCOPE = (2.5, 4.2)
SUBTITLES = [(1.4, 3.9, "Hai il diritto di rimanere in silenzio."), (4.0, 5.3, "Tanto sei già muto.")]
HEAD_SHOT, HEART_SHOT = 5.4, 5.75
DOTS_OFF_AT = np.array([HEAD_SHOT, HEAD_SHOT, HEART_SHOT, HEART_SHOT])
KICKS = ((HEAD_SHOT, 1.0, 0.3), (HEART_SHOT, 0.5, 0.25))
FLASHES = ((HEAD_SHOT, 0.8), (HEART_SHOT, 0.45))


class HandsUpEffect(OneShot):
    DURATION = 9.5

    def setup(self) -> None:
        u = self.unit
        self.glow = radial_glow(int(30 * u), DOT_GLOW, 3)
        self.alarm = (radial_falloff(self.w, self.h, 0.55, 0.8, 1.6)[..., None] * np.array(ALARM, np.float32)).astype(np.uint8)
        self.subtitles = [text_sprite(text, "black", int(42 * u), WHITE, int(5 * u), BLACK) for _, _, text in SUBTITLES]
        self.locked = text_sprite("BERSAGLIO ACQUISITO", "mono", int(20 * u), RED)
        radius = SPLAT_RADIUS * u
        self.splats = {stretch: [splat_sprite(self.rng, radius, stretch) for _ in range(3)] for stretch in SPLAT_STRETCHES}
        self.stain = splat_sprite(self.rng, radius, 1.0, droplets=False)

    def begin(self, scene: Scene) -> None:
        rng = self.rng
        self.snipers = self._snipers_off_screen()
        self.phases = rng.uniform(0, 2 * math.pi, (DOTS, 4))
        self.speeds = rng.uniform(0.6, 1.4, (DOTS, 2))
        self.lock_at = np.sort(rng.uniform(*LOCK, DOTS))
        self.aim_offsets = rng.normal(0, 1, (DOTS, 2))
        self.targets: np.ndarray | None = None
        self.scope_center: np.ndarray | None = None
        self.spatter = LensSpatter(self.w, self.h, self.unit, rng, self.splats)
        self.head: Wound | None = None
        self.chest: Wound | None = None
        self.bullet = np.array([1.0, 0.0])

    def _snipers_off_screen(self) -> list[np.ndarray]:
        w, h = self.w, self.h
        edges = self.rng.permutation(4)
        beyond = 60 * self.unit
        snipers = []
        for i in range(DOTS):
            k = self.rng.uniform(0.1, 0.9)
            left, right, top, bottom = (-beyond, k * h), (w + beyond, k * h), (k * w, -beyond), (k * w, h + beyond)
            snipers.append(np.array((left, right, top, bottom)[edges[i % 4]]))
        return snipers

    def draw(self, frame: np.ndarray, scene: Scene, elapsed: float, dt: float) -> np.ndarray:
        e = elapsed
        self._track_targets(scene.face, scene.body)
        power = envelope(e, self.DURATION, 0.3, 0.8)
        dots = self._dots(e)
        self._shoot_when_due(scene.face, dots, e)
        self._stick_to_body(scene.face, e)
        lights = power * (e < DOTS_OFF_AT).astype(float)
        out = self._red_alert(frame, e, power)
        self._snap_head_back(out, scene.face, e)
        for wound in (self.chest, self.head):
            if wound is not None:
                wound.bleed(dt)
                wound.paste(out)
        self.spatter.fly(out, e, dt)
        self._laser_beams(out, dots, lights)
        self._draw_dots(out, dots, lights, e)
        if SCOPE[0] <= e < SCOPE[1] and self.targets is not None:
            out = self._through_scope(out, e)
        out = self._jolt(out, e)
        self.spatter.cover(out, dt)
        self._flash(out, e)
        self._subtitles(out, e, power)
        return crossfade(frame, out, power)

    def _track_targets(self, face: Face | None, body: Body | None) -> None:
        aim = self._aim(face, body)
        if aim is not None:
            self.targets = ease_toward(self.targets, aim, 0.4)

    def _aim(self, face: Face | None, body: Body | None) -> np.ndarray | None:
        shoulders_seen = body is not None and body.shoulders_confidence > 0.5
        if face is None and not shoulders_seen:
            return None
        if shoulders_seen:
            chest, span, down, heart = self._chest_from_shoulders(body)
        else:
            chest, span, down, heart = self._chest_below_face(face)
        forehead = face.forehead - face.up * 0.15 * face.height if face is not None else chest - down * 0.55 * span
        heart[1] = min(heart[1], self.h - 30 * self.unit)
        return np.array([forehead, forehead, heart, heart])

    @staticmethod
    def _chest_from_shoulders(body: Body) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
        left, right = body.shoulders
        middle, span = (left + right) / 2, np.linalg.norm(left - right)
        down = np.array([-(left - right)[1], (left - right)[0]]) / (span + 1e-6)
        if down[1] < 0:
            down = -down
        heart = middle + (left - middle) * 0.4 + down * 0.3 * span
        return middle, span, down, heart

    @staticmethod
    def _chest_below_face(face: Face) -> tuple[np.ndarray, float, None, np.ndarray]:
        span = 2.4 * face.width
        middle = face.chin - face.up * 0.75 * face.height
        heart_on_image_right = middle + np.array([0.2 * span, 0.3 * span])
        return middle, span, None, heart_on_image_right

    def _dots(self, e: float) -> np.ndarray:
        w, h, u = self.w, self.h, self.unit
        p, s = self.phases, self.speeds
        wander = np.stack(
            [
                w * (0.5 + 0.3 * np.sin(s[:, 0] * 1.7 * e + p[:, 0]) + 0.08 * np.sin(s[:, 0] * 4.1 * e + p[:, 1])),
                h * (0.45 + 0.3 * np.sin(s[:, 1] * 1.3 * e + p[:, 2]) + 0.08 * np.sin(s[:, 1] * 3.7 * e + p[:, 3])),
            ],
            axis=1,
        )
        if self.targets is None:
            return wander
        tremor = 3 * u * np.stack([np.sin(9 * e + p[:, 0]), np.cos(7 * e + p[:, 1])], axis=1)
        aimed = self.targets + self.aim_offsets * 9 * u + tremor
        locked = np.array([smoothstep(t, t + 0.6, e) for t in self.lock_at])[:, None]
        return wander * (1 - locked) + aimed * locked

    def _shoot_when_due(self, face: Face | None, dots: np.ndarray, e: float) -> None:
        if self.head is None and e >= HEAD_SHOT:
            self._shoot_head(face, dots[FOREHEAD_DOT], e)
        if self.chest is None and e >= HEART_SHOT:
            self._shoot_heart(dots[HEART_DOT], e)

    def _shoot_head(self, face: Face | None, aim: np.ndarray, e: float) -> None:
        self.bullet = normalized(aim - self.snipers[FOREHEAD_DOT])
        self.head = self._head_wound(face, aim)
        paint_entry_wound(self.head, self.head.local(aim), self.rng)
        self.spatter.burst(aim, self.bullet, 160, e)
        self._big_splashes_on_lens(aim)

    def _head_wound(self, face: Face | None, aim: np.ndarray) -> Wound:
        if face is not None:
            return Wound(int(2.8 * face.height), face.center, face.up, face.height)
        average_head = 0.3 * self.h
        return Wound(int(2.8 * average_head), aim + (0, 0.5 * average_head), SCREEN_UP, average_head)

    def _big_splashes_on_lens(self, aim: np.ndarray) -> None:
        rng, u = self.rng, self.unit
        for _ in range(7):
            past_the_face = aim + self.bullet * rng.uniform(170, 650) * u + rng.normal(0, 70 * u, 2)
            self.spatter.land(past_the_face, self.bullet * 1500 * u, rng.uniform(0.6, 1.6) * SPLAT_RADIUS * u / 4)

    def _shoot_heart(self, aim: np.ndarray, e: float) -> None:
        rng, u = self.rng, self.unit
        self.chest = Wound(int(0.6 * self.h), aim, SCREEN_UP, 1.0)
        for _ in range(2):
            start = self.chest.center + (rng.normal(0, 8 * u), 30 * u)
            self.chest.drips.add(
                start, rng.uniform(80, 170) * u, rng.uniform(40, 80) * u, rng.uniform(5, 9) * u, rng.uniform(0, 6)
            )
        self.spatter.burst(aim, normalized(aim - self.snipers[HEART_DOT]), 80, e)

    def _stick_to_body(self, face: Face | None, e: float) -> None:
        if self.head is not None and face is not None:
            self.head.follow(face.center, face.up, face.height)
        if self.chest is not None and self.targets is not None:
            self.chest.follow(self.targets[HEART_DOT], SCREEN_UP, 1.0)
            self._soak_shirt(e)

    def _soak_shirt(self, e: float) -> None:
        spread = smoothstep(HEART_SHOT, HEART_SHOT + 1.6, e)
        if spread < 1:
            chest = self.chest
            stamp_onto(chest.canvas, self.stain, chest.center, 0.35 + 1.4 * spread, 0)
            cv2.circle(chest.canvas, pixel(chest.center), max(2, int(6 * self.unit)), BULLET_HOLE, -1, cv2.LINE_AA)

    def _red_alert(self, frame: np.ndarray, e: float, power: float) -> np.ndarray:
        dead = smoothstep(HEAD_SHOT, HEAD_SHOT + 0.8, e)
        pulse = 0.5 + 0.5 * math.sin(2 * math.pi * 1.4 * e)
        out = cv2.addWeighted(frame, 1.0, self.alarm, power * ((0.2 + 0.35 * pulse) * (1 - dead) + 0.6 * dead), 0)
        return desaturate(out, 0.45 * dead * power) if dead > 0 else out

    def _snap_head_back(self, out: np.ndarray, face: Face | None, e: float) -> None:
        since = e - HEAD_SHOT
        if face is None or not 0 <= since < 0.45:
            return
        amount = 34 * self.unit * (since / 0.04 if since < 0.04 else math.exp(-(since - 0.04) / 0.1))
        radius = 0.95 * face.height
        cx, cy = face.center + face.up * 0.1 * face.height
        x0, y0, x1, y1 = clipped_box(int(cx - radius), int(cy - radius), int(cx + radius), int(cy + radius), self.w, self.h)
        if x1 - x0 < 8 or y1 - y0 < 8:
            return
        roi = out[y0:y1, x0:x1]
        dx, dy = self.bullet * amount
        moved = cv2.warpAffine(roi, np.float32([[1, 0, dx], [0, 1, dy]]), (x1 - x0, y1 - y0), borderMode=cv2.BORDER_REFLECT)
        head = np.zeros(roi.shape[:2], np.float32)
        cv2.circle(head, (int(cx - x0), int(cy - y0)), int(radius * 0.85), 1.0, -1, cv2.LINE_AA)
        head = cv2.GaussianBlur(head, (0, 0), radius * 0.12)[..., None]
        roi[:] = (roi * (1 - head) + moved * head).astype(np.uint8)

    def _jolt(self, out: np.ndarray, e: float) -> np.ndarray:
        k = sum(strength * math.exp(-(e - t) / decay) for t, strength, decay in KICKS if e >= t)
        if k < 0.03:
            return out
        dx, dy = self.rng.normal(0, 30 * self.unit * k, 2)
        matrix = cv2.getRotationMatrix2D((self.w / 2, self.h / 2), self.rng.normal(0, 2.2 * k), 1 + 0.05 * k)
        matrix[:, 2] += dx, dy
        return cv2.warpAffine(out, matrix, (self.w, self.h), borderMode=cv2.BORDER_REFLECT)

    def _flash(self, out: np.ndarray, e: float) -> None:
        k = sum(strength * math.exp(-(e - t) / 0.05) for t, strength in FLASHES if e >= t)
        if k > 0.02:
            cv2.addWeighted(out, 1 - k, out, 0, 255 * k, dst=out)

    def _laser_beams(self, out: np.ndarray, dots: np.ndarray, strength: np.ndarray) -> None:
        if strength.max() <= 0:
            return
        layer = np.zeros_like(out)
        for sniper, dot, k in zip(self.snipers, dots, strength):
            if k > 0:
                color = tuple(int(c * k) for c in BEAM)
                cv2.line(layer, pixel(sniper), pixel(dot), color, max(1, int(2 * self.unit)), cv2.LINE_AA)
        cv2.add(out, layer, dst=out)

    def _draw_dots(self, out: np.ndarray, dots: np.ndarray, strength: np.ndarray, e: float) -> None:
        for i, (x, y) in enumerate(dots):
            if strength[i] > 0:
                self._draw_dot(out, int(x), int(y), strength[i] * (0.75 + 0.25 * math.sin(23 * e + 2 * i)))

    def _draw_dot(self, out: np.ndarray, x: int, y: int, brightness: float) -> None:
        add_glow(out, self.glow, (x, y), brightness)
        if 0 <= x < self.w and 0 <= y < self.h:
            core = (np.array(DOT_CORE) * brightness + out[y, x] * (1 - brightness)).astype(int).tolist()
            cv2.circle(out, (x, y), max(2, int(5 * self.unit)), core, -1, cv2.LINE_AA)

    def _through_scope(self, frame: np.ndarray, e: float) -> np.ndarray:
        whole_frame = math.hypot(self.w, self.h)
        closing = smoothstep(SCOPE[0], SCOPE[0] + 0.5, e) * (1 - smoothstep(SCOPE[1] - 0.5, SCOPE[1], e))
        radius = whole_frame + (230 * self.unit - whole_frame) * closing
        self.scope_center = ease_toward(self.scope_center, self.targets[FOREHEAD_DOT], 0.3)
        out = look_through_scope(frame, self.scope_center, radius)
        if closing < 0.3:
            return out
        draw_reticle(out, self.scope_center, radius, self.unit)
        cx, cy = self.scope_center
        if closing > 0.9 and (e * 4) % 1 < 0.7:
            blit(out, self.locked, (cx + radius * 0.45, cy + radius * 0.55), opacity=1.0)
        return out

    def _subtitles(self, out: np.ndarray, e: float, power: float) -> None:
        for (start, end, _), sprite in zip(SUBTITLES, self.subtitles):
            if start <= e < end:
                caption(out, sprite, self.h - 60 * self.unit, power * envelope(e - start, end - start, 0.2, 0.3))
