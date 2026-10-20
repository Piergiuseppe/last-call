import math
import time
from typing import NamedTuple

import cv2
import numpy as np

from ..domain.scene import Scene
from .base import OneShot
from .kit.color import crossfade, grayscale
from .kit.compositing import blit
from .kit.sprites import text_sprite
from .kit.timing import envelope, smoothstep

GREEN = (110, 255, 120)
GREEN_BRIGHT = (210, 255, 210)
GREEN_DIM = (60, 150, 70)
GRID_MINOR = (28, 52, 30)
GRID_MAJOR = (40, 85, 45)
PANEL = (14, 20, 14)
RED = (60, 60, 255)
WHITE = (255, 255, 255)


class Wave(NamedTuple):
    center: float
    amplitude: float
    width: float


P_QRS_T_WAVES = (
    Wave(0.10, 0.12, 0.025),
    Wave(0.20, -0.12, 0.008),
    Wave(0.22, 1.0, 0.010),
    Wave(0.24, -0.28, 0.010),
    Wave(0.42, 0.30, 0.045),
)
R_PEAK = 0.22
RESTING_BPM, PANIC_BPM_RISE = 75, 105

TACHYCARDIA, FIBRILLATION, FLATLINE, DEATH = 2.5, 5.0, 6.5, 7.5


def heartbeat(phase: float) -> float:
    return sum(w.amplitude * math.exp(-(((phase - w.center) / w.width) ** 2)) for w in P_QRS_T_WAVES)


def heart_rate(elapsed: float) -> float:
    return RESTING_BPM + PANIC_BPM_RISE * smoothstep(TACHYCARDIA, FIBRILLATION, elapsed)


def rounded_mask(width: int, height: int, radius: int) -> np.ndarray:
    mask = np.zeros((height, width), np.uint8)
    cv2.rectangle(mask, (radius, 0), (width - radius, height), 255, -1)
    cv2.rectangle(mask, (0, radius), (width, height - radius), 255, -1)
    for x in (radius, width - radius - 1):
        for y in (radius, height - radius - 1):
            cv2.circle(mask, (x, y), radius, 255, -1, cv2.LINE_AA)
    return mask


class HeartMonitorEffect(OneShot):
    DURATION = 10.5
    SWEEP_SECONDS = 2.2
    BLANK_AHEAD = 14

    def setup(self) -> None:
        u = self.unit
        self.pw, self.ph = int(600 * u), int(210 * u)
        self.px, self.py = self.w - self.pw - int(24 * u), self.h - self.ph - int(24 * u)
        self.trace_x, self.trace_w = int(16 * u), int(400 * u)
        self.baseline, self.amplitude = int(140 * u), 85 * u
        self.rate_x = self.trace_x + self.trace_w + int(12 * u)
        self.background = self._panel_background()
        self.panel_alpha = (rounded_mask(self.pw, self.ph, int(14 * u)).astype(np.float32) / 255 * 0.82)[..., None]

    def _panel_background(self) -> np.ndarray:
        u = self.unit
        panel = np.full((self.ph, self.pw, 3), PANEL, np.uint8)
        top, bottom = int(34 * u), self.ph - int(10 * u)
        self._draw_grid(panel, top, bottom)
        cv2.line(panel, (self.rate_x - int(6 * u), int(12 * u)), (self.rate_x - int(6 * u), bottom), GRID_MAJOR, 1)
        label = text_sprite("ECG  II   25 mm/s", "mono", int(16 * u), GREEN_DIM)
        blit(panel, label, (self.trace_x + label.shape[1] / 2, 18 * u))
        hr = text_sprite("HR", "black", int(20 * u), GREEN_DIM)
        blit(panel, hr, (self.rate_x + hr.shape[1] / 2 + 4 * u, 26 * u))
        bpm = text_sprite("bpm", "mono", int(18 * u), GREEN_DIM)
        blit(panel, bpm, ((self.rate_x + self.pw) / 2, self.ph - 28 * u))
        return panel

    def _draw_grid(self, panel: np.ndarray, top: int, bottom: int) -> None:
        x0, x1 = self.trace_x, self.trace_x + self.trace_w
        step = 10 * self.unit
        for i in range(int(self.trace_w / step) + 1):
            x = int(x0 + i * step)
            cv2.line(panel, (x, top), (x, bottom), GRID_MAJOR if i % 5 == 0 else GRID_MINOR, 1)
        for i in range(int((bottom - top) / step) + 1):
            y = int(top + i * step)
            cv2.line(panel, (x0, y), (x1, y), GRID_MAJOR if i % 5 == 0 else GRID_MINOR, 1)

    def begin(self, scene: Scene) -> None:
        self.values = np.full(self.trace_w, np.nan, np.float32)
        self.head = 0.0
        self.phase = 0.0
        self.last_beat = -10.0
        self.clock = time.strftime("%H:%M")
        self.fibrillation_phases = self.rng.uniform(0, 2 * math.pi, 3)
        self.shown_bpm = RESTING_BPM

    def _sample(self, s: float, step: float) -> float:
        noise = self.rng.normal(0, 0.012)
        if s < FIBRILLATION:
            return self._sinus_rhythm(s, step) + noise
        if s < FLATLINE:
            return self._fibrillation(s) + 4 * noise
        return noise * 0.3

    def _sinus_rhythm(self, s: float, step: float) -> float:
        previous, self.phase = self.phase, self.phase + heart_rate(s) / 60 * step
        if previous < R_PEAK <= self.phase:
            self.last_beat = s
        self.phase %= 1.0
        return heartbeat(self.phase)

    def _fibrillation(self, s: float) -> float:
        p = self.fibrillation_phases
        fading = 1 - 0.6 * smoothstep(FIBRILLATION, FLATLINE, s)
        wave = 0.4 * math.sin(2 * math.pi * 5.3 * s + p[0]) + 0.25 * math.sin(2 * math.pi * 8.1 * s + p[1])
        wave += 0.15 * math.sin(2 * math.pi * 12.7 * s + p[2])
        return fading * wave

    def _advance(self, elapsed: float, dt: float) -> None:
        samples_per_second = self.trace_w / self.SWEEP_SECONDS
        target = self.head + samples_per_second * dt
        for idx in range(int(self.head) + 1, int(target) + 1):
            when = elapsed - (target - idx) / samples_per_second
            self.values[idx % self.trace_w] = self._sample(when, 1 / samples_per_second)
        self.head = target
        start = int(self.head) + 1
        self.values[np.arange(start, start + self.BLANK_AHEAD) % self.trace_w] = np.nan

    def _draw_trace(self, panel: np.ndarray) -> None:
        u = self.unit
        head = int(self.head) % self.trace_w
        xs = np.arange(self.trace_w) + self.trace_x
        ys = self.baseline - self.values * self.amplitude
        glow = np.zeros_like(panel)
        segments = []
        for lo, hi in ((0, head + 1), (head + 1, self.trace_w)):
            drawn = np.isfinite(ys[lo:hi])
            if drawn.sum() > 1:
                segments.append(np.stack([xs[lo:hi][drawn], ys[lo:hi][drawn]], axis=1).astype(np.int32))
        cv2.polylines(glow, segments, False, GREEN, max(2, int(6 * u)), cv2.LINE_AA)
        cv2.add(panel, cv2.GaussianBlur(glow, (0, 0), 5 * u), dst=panel)
        cv2.polylines(panel, segments, False, GREEN_BRIGHT, max(1, int(2 * u)), cv2.LINE_AA)
        if np.isfinite(ys[head]):
            cv2.circle(panel, (int(xs[head]), int(ys[head])), max(2, int(4 * u)), WHITE, -1, cv2.LINE_AA)

    def draw(self, frame: np.ndarray, scene: Scene, elapsed: float, dt: float) -> np.ndarray:
        self._advance(elapsed, dt)
        blink = (elapsed * 3) % 1 < 0.6
        out = self._drain_color(frame, smoothstep(FLATLINE, FLATLINE + 1.2, elapsed))
        self._draw_panel(out, elapsed, blink)
        alarm = self._alarm(elapsed, blink)
        if alarm is not None:
            blit(out, alarm, (self.px + self.pw - alarm.shape[1] / 2, self.py - alarm.shape[0] / 2 - 6 * self.unit))
        if elapsed >= DEATH:
            self._draw_time_of_death(out, elapsed)
        return crossfade(frame, out, envelope(elapsed, self.DURATION, 0.3, 0.8))

    @staticmethod
    def _drain_color(frame: np.ndarray, dead: float) -> np.ndarray:
        if dead <= 0:
            return frame.copy()
        return cv2.addWeighted(frame, 1 - 0.85 * dead, grayscale(frame), 0.85 * dead * (1 - 0.2 * dead), 0)

    def _draw_panel(self, out: np.ndarray, elapsed: float, blink: bool) -> None:
        panel = self.background.copy()
        self._draw_trace(panel)
        self._draw_rate(panel, elapsed, blink)
        if elapsed >= TACHYCARDIA and blink:
            cv2.rectangle(panel, (1, 1), (self.pw - 2, self.ph - 2), RED, max(2, int(3 * self.unit)))
        roi = out[self.py : self.py + self.ph, self.px : self.px + self.pw]
        roi[:] = (roi * (1 - self.panel_alpha) + panel * self.panel_alpha).astype(np.uint8)

    def _alarm(self, elapsed: float, blink: bool) -> np.ndarray | None:
        u = self.unit
        if TACHYCARDIA <= elapsed < FIBRILLATION and blink:
            return text_sprite("TACHICARDIA", "black", int(40 * u), RED, int(3 * u))
        if FIBRILLATION <= elapsed < FLATLINE and (elapsed * 4) % 1 < 0.6:
            return text_sprite("FIBRILLAZIONE", "black", int(40 * u), RED, int(3 * u))
        if elapsed >= FLATLINE:
            letters = min(18, 4 + int((elapsed - FLATLINE) * 12))
            return text_sprite("B" + "I" * letters + "P", "black", int(44 * u), RED, int(3 * u))
        return None

    def _draw_rate(self, panel: np.ndarray, elapsed: float, blink: bool) -> None:
        u = self.unit
        since_beat = elapsed - self.last_beat
        color = self._update_shown_rate(elapsed, blink)
        number = text_sprite(str(self.shown_bpm), "black", int(64 * u), color)
        blit(panel, number, ((self.rate_x + self.pw) / 2, self.ph * 0.56))
        pulse = 1 + 0.35 * math.exp(-since_beat / 0.12) if elapsed < FIBRILLATION else 1.0
        heart = text_sprite("♥", "mono", int(34 * u), RED if elapsed >= TACHYCARDIA else GREEN)
        blit(panel, heart, (self.pw - 26 * u, 28 * u), pulse)
        if elapsed < FIBRILLATION and since_beat < 0.35:
            beep = text_sprite("BIP", "black", int(20 * u), GREEN_BRIGHT)
            blit(panel, beep, (self.pw - 70 * u, 28 * u), opacity=1 - since_beat / 0.35)

    def _update_shown_rate(self, elapsed: float, blink: bool) -> tuple[int, int, int]:
        if elapsed < FIBRILLATION:
            self.shown_bpm = int(heart_rate(elapsed))
            return GREEN
        if elapsed < FLATLINE:
            erratic = self.rng.random() < 0.25
            if erratic:
                self.shown_bpm = int(self.rng.integers(30, 320))
            return RED
        self.shown_bpm = 0
        return RED if blink else GREEN_DIM

    def _draw_time_of_death(self, out: np.ndarray, elapsed: float) -> None:
        u = self.unit
        appear = smoothstep(DEATH, DEATH + 0.4, elapsed)
        band_h = int(110 * u)
        y0 = int(self.h * 0.1)
        band = out[y0 : y0 + band_h]
        band[:] = (band * (1 - 0.6 * appear)).astype(np.uint8)
        text = text_sprite(f"ORA DEL DECESSO: {self.clock}", "black", int(56 * u), WHITE, int(4 * u), (0, 0, 0))
        blit(out, text, (self.w / 2, y0 + band_h / 2), 1.15 - 0.15 * appear, opacity=appear)
