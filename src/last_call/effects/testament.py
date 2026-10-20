import math
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ..domain.scene import Scene
from .base import OneShot
from .kit.color import crossfade
from .kit.compositing import blit, shake
from .kit.sprites import FONTS, emoji_sprite, premultiply, text_sprite, wrap_text
from .kit.timing import envelope, smoothstep

QUILL = "\U0001fab6"
QUILL_NIB = 0.42
PARCHMENT = (172, 212, 232)
BURNT = (22, 40, 70)
INK = (28, 36, 58)
STAMP_PAPER, STAMP_INK, STAMP_FIELD = (236, 240, 242), (120, 60, 120), (205, 175, 225)
WAX, WAX_DARK, WAX_SHINE = (40, 35, 175), (25, 20, 115), (110, 110, 235)
SEPIA = np.array([[0.131, 0.534, 0.272], [0.168, 0.686, 0.349], [0.189, 0.769, 0.393]], np.float32)
BULLET = "•  "

BEQUESTS = [
    "il mio debito tecnico a chi resta",
    "la password del Wi‑Fi a nessuno: me la porto nella tomba",
    "le riunioni del venerdì pomeriggio a tutti voi, con affetto",
    "il mio posto auto al primo che arriva, tanto lo fate già",
    "la mia tazza a nessuno: chi la tocca, lo vengo a trovare di notte",
    "i miei commit senza test al team QA",
    "le ferie non godute all'azienda, come sempre",
    "i miei 4.000 messaggi non letti in chat al mio successore",
    "la mia sedia ergonomica al collega col mal di schiena (sai chi sei)",
    "il deploy del venerdì sera a chi ha detto «tanto è una modifica piccola»",
    "il mio badge al tornello, che tanto non mi ha mai riconosciuto",
    "le mie riunioni ricorrenti: continuate pure, tanto non se ne accorge nessuno",
    "il mio microfono acceso per sbaglio a chi parla sempre sopra gli altri",
    "le password sul post‑it sotto la tastiera a chiunque le trovi",
]

UNROLL, TITLE, WRITE, SIGN, STAMP, SEAL = (0.15, 1.2), 1.25, 1.9, 6.0, 7.0, 7.3
WRITING_SPEED = 950
SIGNING_SPEED = 560
INK_DRYING = 12


def _blotches(shape: tuple[int, int], cell: int, rng: np.random.Generator) -> np.ndarray:
    coarse = rng.random((max(2, shape[0] // cell), max(2, shape[1] // cell))).astype(np.float32)
    return cv2.GaussianBlur(cv2.resize(coarse, shape[::-1], interpolation=cv2.INTER_CUBIC), (0, 0), cell / 3)


def _text_mask(text: str, font: str, size: int) -> np.ndarray:
    face = ImageFont.truetype(FONTS[font], size)
    left, top, right, bottom = face.getbbox(text)
    image = Image.new("L", (int(right - left) + 4, int(bottom - top) + 4))
    ImageDraw.Draw(image).text((2 - left, 2 - top), text, font=face, fill=255)
    return np.asarray(image, np.float32) / 255


@dataclass(frozen=True)
class Line:
    text: str
    font: str
    size: int
    y: int
    height: int
    indent: int = 0
    title: bool = False

    @property
    def bottom(self) -> int:
        return self.y + self.height


@dataclass(frozen=True)
class Stroke:
    start: float
    end: float
    x0: int
    x1: int
    y: int


class Manuscript:
    def __init__(self, height: int, width: int) -> None:
        self.cover = np.zeros((height, width), np.float32)
        self.written_at = np.full((height, width), np.inf, np.float32)
        self.strokes: list[Stroke] = []

    def write(self, mask: np.ndarray, x: int, y: int, start: float, speed: float) -> float:
        h, w = mask.shape
        h, w = min(h, self.cover.shape[0] - y), min(w, self.cover.shape[1] - x)
        if h <= 0 or w <= 0:
            return start
        mask = mask[:h, :w]
        end = start + w / speed
        ink = mask > 0.02
        left_to_right = start + np.arange(w, dtype=np.float32)[None, :] / speed
        region = self.written_at[y : y + h, x : x + w]
        region[ink] = np.broadcast_to(left_to_right, (h, w))[ink]
        self.cover[y : y + h, x : x + w] = np.maximum(self.cover[y : y + h, x : x + w], mask)
        inked_rows = np.nonzero(ink.any(axis=1))[0]
        middle = y + (int(inked_rows.mean()) if len(inked_rows) else h // 2)
        self.strokes.append(Stroke(start, end, x, x + w, middle))
        return end

    def ink(self, elapsed: float, rows: int) -> np.ndarray:
        return self.cover[:rows] * np.clip((elapsed - self.written_at[:rows]) * INK_DRYING, 0, 1)

    def pen(self, elapsed: float) -> tuple[float, int] | None:
        for s in self.strokes:
            if s.start <= elapsed < s.end:
                return s.x0 + (s.x1 - s.x0) * (elapsed - s.start) / (s.end - s.start), s.y
        return None


class TestamentEffect(OneShot):
    DURATION = 10.0

    def setup(self) -> None:
        u = self.unit
        self.pw, self.ph = int(500 * u), int(640 * u)
        self.y0 = int(40 * u)
        self.paper, self.paper_alpha = self._parchment()
        self.ink = np.full_like(self.paper, INK)
        self.roll = self._roll()
        self.revenue_stamp = self._revenue_stamp()
        self.quill = emoji_sprite(QUILL)

    def begin(self, scene: Scene) -> None:
        self.x0 = self._away_from_face(scene)
        picks = self.rng.choice(len(BEQUESTS), 3, replace=False)
        self.manuscript, self.signed = self._write(_punctuated([BEQUESTS[i] for i in picks]))
        self.seal = self._seal(self.name.strip()[:1].upper() or "X")

    def _away_from_face(self, scene: Scene) -> int:
        margin = int(50 * self.unit)
        face_on_left = scene.face is None or scene.face.center[0] < self.w / 2
        return self.w - self.pw - margin if face_on_left else margin

    def draw(self, frame: np.ndarray, scene: Scene, elapsed: float, dt: float) -> np.ndarray:
        e = elapsed
        power = envelope(e, self.DURATION, 0.3, 0.8)
        out = cv2.addWeighted(frame, 1 - 0.6 * power, cv2.transform(frame, SEPIA), 0.6 * power, 0)
        self._draw_parchment(out, e)
        thud = self._draw_stamp_and_seal(out, e)
        self._draw_quill(out, e)
        if thud > 0:
            out = shake(out, 6 * self.unit * thud, self.rng)
        return crossfade(frame, out, power)

    def _draw_parchment(self, out: np.ndarray, e: float) -> None:
        unrolled = int(self.ph * smoothstep(*UNROLL, e))
        x0, y0 = self.x0, self.y0
        if unrolled > 2:
            written = self.manuscript.ink(e, unrolled)
            page = cv2.blendLinear(self.ink[:unrolled], self.paper[:unrolled], written, 1 - written)
            alpha = self.paper_alpha[:unrolled]
            roi = out[y0 : y0 + unrolled, x0 : x0 + self.pw]
            roi[:] = cv2.blendLinear(page, roi, alpha, 1 - alpha)
        cx = x0 + self.pw / 2
        blit(out, self.roll, (cx, y0))
        blit(out, self.roll, (cx, y0 + unrolled))

    def _draw_stamp_and_seal(self, out: np.ndarray, e: float) -> float:
        u = self.unit
        wait_for_signature = max(0.0, self.signed + 0.3 - STAMP)
        thud = 0.0
        for start, sprite, (sx, sy), angle in (
            (STAMP + wait_for_signature, self.revenue_stamp, (self.pw - 92 * u, self.ph - 108 * u), -7),
            (SEAL + wait_for_signature, self.seal, (self.pw * 0.5 + 10 * u, self.ph - 62 * u), 12),
        ):
            if e >= start:
                land = smoothstep(start, start + 0.12, e)
                blit(out, sprite, (self.x0 + sx, self.y0 + sy), 1 + 0.6 * (1 - land), angle, land)
                thud = max(thud, 1 - smoothstep(start + 0.1, start + 0.35, e))
        return thud

    def _draw_quill(self, out: np.ndarray, e: float) -> None:
        pen = self.manuscript.pen(e)
        if pen is None:
            return
        px, py = pen
        size = 90 * self.unit
        wobble = 4 * self.unit * math.sin(e * 40)
        nib = size * QUILL_NIB
        blit(out, self.quill, (self.x0 + px + nib, self.y0 + py - nib + wobble), size / self.quill.shape[0])

    def _write(self, bequests: list[str]) -> tuple[Manuscript, float]:
        u = self.unit
        margin = int(46 * u)
        signatures_top = self.ph - int(195 * u)
        for scale in (1.0, 0.94, 0.88, 0.82, 0.76, 0.7, 0.64):
            lines = self._lines(bequests, scale, self.pw - 2 * margin)
            if lines[-1].bottom <= signatures_top:
                break
        manuscript = Manuscript(self.ph, self.pw)
        t = TITLE
        for line in lines:
            t = TITLE if line.title else max(t, WRITE)
            mask = _text_mask(line.text, line.font, line.size)
            t = manuscript.write(mask, margin + line.indent, line.y, t, WRITING_SPEED * u) + 0.12
        return manuscript, self._sign(manuscript, margin, t)

    def _sign(self, manuscript: Manuscript, margin: int, t: float) -> float:
        u = self.unit
        in_faith = _text_mask("In fede,", "serif_italic", int(28 * u))
        t = manuscript.write(in_faith, margin, self.ph - int(180 * u), max(t, SIGN), WRITING_SPEED * u) + 0.15
        signature = _text_mask(self.name, "signature", int(64 * u))
        limit = int(250 * u)
        if signature.shape[1] > limit:
            height = max(1, signature.shape[0] * limit // signature.shape[1])
            signature = cv2.resize(signature, (limit, height), interpolation=cv2.INTER_AREA)
        sx, sy = int(margin + 20 * u), self.ph - int(150 * u)
        t = manuscript.write(signature, sx, sy, t, SIGNING_SPEED * u)
        flourish = self._flourish(max(signature.shape[1], int(140 * u)))
        return manuscript.write(flourish, sx - int(10 * u), sy + signature.shape[0] - int(6 * u), t, 2 * SIGNING_SPEED * u)

    def _lines(self, bequests: list[str], scale: float, width: int) -> list[Line]:
        u = self.unit * scale
        body = int(31 * u)
        pitch = int(body * 1.28)
        bullet = int(ImageFont.truetype(FONTS["serif_italic"], body).getlength(BULLET))
        title = Line("Testamento", "gothic", int(70 * u), int(28 * self.unit), int(80 * u), title=True)
        lines = [title]
        y = title.bottom + int(14 * u)
        intro = f"Io sottoscritto {self.name}, nel pieno delle mie facoltà mentali (più o meno), lascio:"
        for text in wrap_text(intro, "serif_italic", body, width):
            lines.append(Line(text, "serif_italic", body, y, pitch))
            y += pitch
        y += int(10 * u)
        for item in bequests:
            for k, text in enumerate(wrap_text(item, "serif_italic", body, width - bullet)):
                first = k == 0
                lines.append(Line((BULLET if first else "") + text, "serif_italic", body, y, pitch, 0 if first else bullet))
                y += pitch
            y += int(8 * u)
        return lines

    def _flourish(self, width: int) -> np.ndarray:
        u = self.unit
        h = int(30 * u)
        mask = np.zeros((h, width + int(30 * u)), np.float32)
        s = np.linspace(0, 1, 120)
        xs = s * width + 22 * u * np.sin(s * math.pi) ** 8 * np.cos(s * 9 * math.pi)
        ys = h / 2 + 6 * u * np.sin(s * 2.5 * math.pi) * (1 - s) + 9 * u * np.sin(s * 9 * math.pi) * s**6
        pts = np.stack([xs + 4 * u, ys], axis=1).astype(np.int32)
        cv2.polylines(mask, [pts], False, 1.0, max(2, int(2.5 * u)), cv2.LINE_AA)
        return mask

    def _parchment(self) -> tuple[np.ndarray, np.ndarray]:
        u, pw, ph = self.unit, self.pw, self.ph
        stains = _blotches((ph, pw), int(60 * u), self.rng) - 0.5
        grain = self.rng.normal(0, 5, (ph, pw)).astype(np.float32)
        paper = np.array(PARCHMENT, np.float32) + (stains * 60 + grain)[..., None]
        yy, xx = np.mgrid[0:ph, 0:pw].astype(np.float32)
        from_edge = np.minimum(np.minimum(xx, pw - 1 - xx), np.minimum(yy, ph - 1 - yy))
        from_edge -= 14 * u * _blotches((ph, pw), int(18 * u), self.rng)
        alpha = np.clip(from_edge / 1.5, 0, 1)
        burn = np.clip(1 - from_edge / (22 * u), 0, 1) ** 1.6
        paper = paper * (1 - burn[..., None]) + np.array(BURNT, np.float32) * burn[..., None]
        paper *= (1 - 0.22 * np.clip(1 - from_edge / (90 * u), 0, 1))[..., None]
        return np.clip(paper, 0, 255).astype(np.uint8), alpha.astype(np.float32)

    def _roll(self) -> np.ndarray:
        u = self.unit
        rh, knob = int(34 * u), int(16 * u)
        width = self.pw + int(8 * u) + 2 * knob
        sprite = np.zeros((rh + int(10 * u), width, 4), np.uint8)
        top = int(5 * u)
        shade = 0.5 + 0.5 * np.sin(np.linspace(0, math.pi, rh))[:, None]
        paper = (np.array(PARCHMENT, np.float32) * 0.92 * shade[..., None]).astype(np.uint8)
        sprite[top : top + rh, knob : width - knob, :3] = paper
        sprite[top : top + rh, knob : width - knob, 3] = 255
        for cx in (knob // 2 + 1, width - knob // 2 - 2):
            cy = top + rh // 2
            cv2.ellipse(sprite, (cx, cy), (knob // 2 + 2, rh // 2 + int(4 * u)), 0, 0, 360, (30, 60, 100, 255), -1, cv2.LINE_AA)
            highlight = (max(1, knob // 6), int(5 * u))
            cv2.ellipse(sprite, (cx - 2, cy - int(6 * u)), highlight, 0, 0, 360, (90, 130, 170, 255), -1, cv2.LINE_AA)
        return sprite

    def _revenue_stamp(self) -> np.ndarray:
        u = self.unit
        w, h = int(128 * u), int(158 * u)
        stamp = np.zeros((h, w, 4), np.uint8)
        stamp[...] = (*STAMP_PAPER, 255)
        pad = int(11 * u)
        field = stamp[pad : h - pad, pad : w - pad]
        field[...] = (*STAMP_FIELD, 255)
        self._guilloche(field)
        fh, fw = field.shape[:2]
        for text, font, px, y in (
            ("MARCA DA BOLLO", "black", 12, 14 * u),
            ("€16,00", "black", 30, fh / 2 + 6 * u),
            ("SEDICI/00", "mono", 12, fh - 13 * u),
        ):
            sprite = text_sprite(text, font, int(px * u), STAMP_INK)
            blit(field[..., :3], sprite, (fw / 2, y), min(1.0, (fw - 10 * u) / sprite.shape[1]))
        cv2.circle(field, (fw // 2, int(fh * 0.3)), int(10 * u), (*STAMP_INK, 255), max(1, int(2 * u)), cv2.LINE_AA)
        self._perforate(stamp)
        return premultiply(stamp)

    def _guilloche(self, field: np.ndarray) -> None:
        fh, fw = field.shape[:2]
        color = (*[int(c * 0.85) for c in STAMP_FIELD], 255)
        for k in range(14):
            ys = (k + 0.5) * fh / 14 + 3 * self.unit * np.sin(np.linspace(0, 6 * math.pi, fw) + k)
            pts = np.stack([np.arange(fw), ys], axis=1).astype(np.int32)
            cv2.polylines(field, [pts], False, color, 1, cv2.LINE_AA)

    def _perforate(self, stamp: np.ndarray) -> None:
        h, w = stamp.shape[:2]
        r, step = max(2, int(4 * self.unit)), int(11 * self.unit)
        for x in range(step // 2, w, step):
            for y in (0, h - 1):
                cv2.circle(stamp, (x, y), r, (0, 0, 0, 0), -1)
        for y in range(step // 2, h, step):
            for x in (0, w - 1):
                cv2.circle(stamp, (x, y), r, (0, 0, 0, 0), -1)

    def _seal(self, letter: str) -> np.ndarray:
        u = self.unit
        r = int(46 * u)
        size = 2 * r + int(12 * u)
        seal = np.zeros((size, size, 4), np.uint8)
        c = size / 2
        angles = np.linspace(0, 2 * math.pi, 40, endpoint=False)
        poured = r * (1 + 0.08 * self.rng.standard_normal(40))
        blob = np.stack([c + poured * np.cos(angles), c + poured * np.sin(angles)], axis=1).astype(np.int32)
        cv2.fillPoly(seal, [blob], (*WAX, 255), cv2.LINE_AA)
        cv2.circle(seal, (int(c), int(c)), int(0.68 * r), (*WAX_DARK, 255), max(2, int(4 * u)), cv2.LINE_AA)
        blit(seal[..., :3], text_sprite(letter, "serif_bold", int(52 * u), WAX_DARK), (c, c))
        shine_axes = (int(0.22 * r), int(0.1 * r))
        cv2.ellipse(seal, (int(c - 0.35 * r), int(c - 0.4 * r)), shine_axes, -35, 0, 360, (*WAX_SHINE, 255), -1, cv2.LINE_AA)
        return seal


def _punctuated(items: list[str]) -> list[str]:
    return [item + (";" if k < len(items) - 1 else ".") for k, item in enumerate(items)]
