import time
from functools import cache

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ..domain.scene import Scene
from .base import OneShot
from .kit.color import crossfade
from .kit.compositing import add_glow, blit, radial_glow
from .kit.sprites import FONTS, emoji_sprite, premultiply, wrap_text
from .kit.timing import ease_toward, envelope

PAPER = (222, 233, 238)
INK = (22, 22, 22)
CANDLE_LIGHT = (60, 170, 255)
CANDLE = "\U0001f56f️"
FLOWERS = "\U0001f490"
LINE_PITCH = 1.15
TEXT_INSET = 20
SHRINK_STEPS = (1.0, 0.96, 0.92, 0.88, 0.84, 0.8, 0.75, 0.7)


@cache
def _paragraph(lines: tuple[str, ...], font: str, size: int) -> np.ndarray:
    face = ImageFont.truetype(FONTS[font], size)
    ascent, descent = face.getmetrics()
    pitch = int(size * LINE_PITCH)
    pad = 6
    width = int(max(face.getlength(line) for line in lines)) + 2 * pad
    image = Image.new("RGBA", (width, pitch * (len(lines) - 1) + ascent + descent + 2 * pad))
    draw = ImageDraw.Draw(image)
    for i, line in enumerate(lines):
        draw.text((width / 2, pad + i * pitch), line, font=face, fill=INK[::-1], anchor="ma")
    return premultiply(cv2.cvtColor(np.array(image), cv2.COLOR_RGBA2BGRA))


class FuneralEffect(OneShot):
    DURATION = 9.0

    def setup(self) -> None:
        u = self.unit
        self.photo = (int(90 * u), int(110 * u), int(410 * u), int(540 * u))
        self.text_box = (int(470 * u), int(115 * u), int(1215 * u), int(665 * u))
        self.candles = [(int(115 * u), int(612 * u)), (int(385 * u), int(612 * u))]
        self.glow = radial_glow(int(46 * u), CANDLE_LIGHT, 2.5)
        self.photo_crop: np.ndarray | None = None
        self.flicker = np.ones(len(self.candles))

    def begin(self, scene: Scene) -> None:
        self.poster = self._poster(time.strftime("%H:%M"))
        self.photo_crop = None

    def draw(self, frame: np.ndarray, scene: Scene, elapsed: float, dt: float) -> np.ndarray:
        out = self.poster.copy()
        self._draw_photo(out, frame, scene)
        self._draw_flames(out)
        return crossfade(frame, out, envelope(elapsed, self.DURATION, 0.6, 0.8))

    def _draw_photo(self, out: np.ndarray, frame: np.ndarray, scene: Scene) -> None:
        x0, y0, x1, y1 = self.photo
        pw, ph = x1 - x0, y1 - y0
        self.photo_crop = ease_toward(self.photo_crop, self._crop_around_face(scene), 0.2)
        cx, cy, ch = self.photo_crop
        ch = min(ch, self.h, self.w * ph / pw)
        cw = ch * pw / ph
        left = int(np.clip(cx - cw / 2, 0, self.w - cw))
        top = int(np.clip(cy - ch / 2, 0, self.h - ch))
        part = frame[top : top + int(ch), left : left + int(cw)]
        gray = cv2.cvtColor(cv2.resize(part, (pw, ph), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
        gray = cv2.convertScaleAbs(gray, alpha=1.15, beta=-12)
        out[y0:y1, x0:x1] = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        self._draw_mourning_ribbon(out)

    def _crop_around_face(self, scene: Scene) -> np.ndarray:
        face = scene.face
        if face is None:
            return np.array([self.w / 2, self.h / 2, float(self.h)])
        return np.array([face.center[0], face.center[1] + 0.25 * face.height, 2.7 * face.height])

    def _draw_mourning_ribbon(self, out: np.ndarray) -> None:
        _, y0, x1, _ = self.photo
        band = int(34 * self.unit)
        corner = np.array([[x1 - 3 * band, y0], [x1 - 2 * band, y0], [x1, y0 + 2 * band], [x1, y0 + 3 * band]], np.int32)
        cv2.fillPoly(out, [corner], INK, cv2.LINE_AA)

    def _draw_flames(self, out: np.ndarray) -> None:
        self.flicker = ease_toward(self.flicker, self.rng.uniform(0.45, 1.0, len(self.candles)), 0.4)
        above_wick = int(40 * self.unit)
        for (x, y), brightness in zip(self.candles, self.flicker):
            add_glow(out, self.glow, (x, y - above_wick), brightness)

    def _poster(self, hour: str) -> np.ndarray:
        poster = self._paper()
        self._draw_borders(poster)
        self._draw_cross(poster)
        self._draw_photo_frame(poster)
        self._draw_candles_and_flowers(poster)
        self._draw_text(poster, hour)
        return poster

    def _paper(self) -> np.ndarray:
        w, h = self.w, self.h
        noise = cv2.resize(self.rng.normal(0, 5, (h // 8, w // 8)).astype(np.float32), (w, h))
        return np.clip(np.array(PAPER, np.float32) + noise[..., None], 0, 255).astype(np.uint8)

    def _draw_borders(self, poster: np.ndarray) -> None:
        u, w, h = self.unit, self.w, self.h
        cv2.rectangle(poster, (int(14 * u), int(14 * u)), (w - int(14 * u), h - int(14 * u)), INK, int(14 * u))
        cv2.rectangle(poster, (int(34 * u), int(34 * u)), (w - int(34 * u), h - int(34 * u)), INK, max(1, int(2 * u)))

    def _draw_cross(self, poster: np.ndarray) -> None:
        u = self.unit
        cx = (self.text_box[0] + self.text_box[2]) // 2
        bar = max(2, int(6 * u))
        cv2.line(poster, (cx, int(46 * u)), (cx, int(100 * u)), INK, bar)
        cv2.line(poster, (cx - int(17 * u), int(62 * u)), (cx + int(17 * u), int(62 * u)), INK, bar)

    def _draw_photo_frame(self, poster: np.ndarray) -> None:
        u = self.unit
        x0, y0, x1, y1 = self.photo
        pad = int(8 * u)
        cv2.rectangle(poster, (x0 - pad, y0 - pad), (x1 + pad, y1 + pad), INK, max(2, int(5 * u)))

    def _draw_candles_and_flowers(self, poster: np.ndarray) -> None:
        u = self.unit
        candle, flowers = emoji_sprite(CANDLE), emoji_sprite(FLOWERS)
        for x, y in self.candles:
            blit(poster, candle, (x, y + int(8 * u)), 95 * u / candle.shape[0])
        between = (self.candles[0][0] + self.candles[1][0]) / 2
        blit(poster, flowers, (between, self.candles[0][1] + 6 * u), 105 * u / flowers.shape[0])

    def _draw_text(self, poster: np.ndarray, hour: str) -> None:
        x0, y0, x1, y1 = self.text_box
        blocks, total = self._fitting_blocks(hour, x1 - x0, y1 - y0)
        y = y0 + (y1 - y0 - total) // 2
        for sprite, gap in blocks:
            blit(poster, sprite, ((x0 + x1) / 2, y + sprite.shape[0] / 2))
            y += sprite.shape[0] + gap

    def _fitting_blocks(self, hour: str, width: int, height: int) -> tuple[list[tuple[np.ndarray, int]], int]:
        for scale in SHRINK_STEPS:
            blocks = self._text_blocks(hour, width, scale)
            total = sum(s.shape[0] + gap for s, gap in blocks)
            if total <= height and max(s.shape[1] for s, _ in blocks) <= width:
                break
        return blocks, total

    def _text_blocks(self, hour: str, width: int, scale: float) -> list[tuple[np.ndarray, int]]:
        u = self.unit * scale

        def text(content: str, font: str, size: float) -> np.ndarray:
            px = int(size * u)
            return _paragraph(tuple(wrap_text(content, font, px, width - TEXT_INSET)), font, px)

        name = self.name.upper()
        rule = np.zeros((max(2, int(3 * u)), int(220 * u), 4), np.uint8)
        rule[..., :3], rule[..., 3] = INK, 255
        return [
            (text("È venuto a mancare all'affetto dei suoi cari", "serif_italic", 40), int(10 * u)),
            (text(name, "serif_bold", self._name_size(name, u, width)), int(6 * u)),
            (text("di anni: troppi", "serif_italic", 38), int(18 * u)),
            (rule, int(18 * u)),
            (
                text(
                    "Ne danno il triste annuncio gli amici della videochiamata, che non ne sentiranno la mancanza.", "serif", 35
                ),
                int(16 * u),
            ),
            (text(f"Ora del decesso: {hour}", "serif_bold", 42), int(16 * u)),
            (text("Non fiori ma opere di bene (tipo spegnere la webcam).", "serif_italic", 33), int(12 * u)),
            (text("Le esequie si terranno su questa chiamata, appena qualcuno smette di ridere.", "serif", 33), 0),
        ]

    @staticmethod
    def _name_size(name: str, u: float, width: int) -> float:
        size = 108.0
        length = ImageFont.truetype(FONTS["serif_bold"], int(size * u)).getlength(name)
        if length > width - TEXT_INSET:
            size *= (width - TEXT_INSET) / length
        return size
