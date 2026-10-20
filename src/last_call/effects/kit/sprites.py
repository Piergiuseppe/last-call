import math
from functools import cache

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

_SUPPLEMENTAL = "/System/Library/Fonts/Supplemental/"
FONTS = {
    "serif": _SUPPLEMENTAL + "Times New Roman.ttf",
    "serif_bold": _SUPPLEMENTAL + "Times New Roman Bold.ttf",
    "serif_italic": _SUPPLEMENTAL + "Times New Roman Italic.ttf",
    "didot": _SUPPLEMENTAL + "Didot.ttc",
    "gothic": _SUPPLEMENTAL + "Trattatello.ttf",
    "impact": _SUPPLEMENTAL + "Impact.ttf",
    "black": _SUPPLEMENTAL + "Arial Black.ttf",
    "signature": _SUPPLEMENTAL + "SnellRoundhand.ttc",
    "mono": "/System/Library/Fonts/Menlo.ttc",
}
EMOJI_FONT = "/System/Library/Fonts/Apple Color Emoji.ttc"
EMOJI_SIZE = 160


def premultiply(bgra: np.ndarray) -> np.ndarray:
    out = bgra.copy()
    out[..., :3] = (bgra[..., :3].astype(np.float32) * bgra[..., 3:] / 255).astype(np.uint8)
    return out


@cache
def emoji_sprite(char: str) -> np.ndarray:
    font = ImageFont.truetype(EMOJI_FONT, EMOJI_SIZE)
    image = Image.new("RGBA", (260, 260))
    ImageDraw.Draw(image).text((130, 130), char, font=font, embedded_color=True, anchor="mm")
    return premultiply(cv2.cvtColor(np.array(image.crop(image.getbbox())), cv2.COLOR_RGBA2BGRA))


@cache
def text_sprite(
    text: str,
    font: str,
    size: int,
    color: tuple[int, int, int] = (255, 255, 255),
    stroke: int = 0,
    stroke_color: tuple[int, int, int] = (0, 0, 0),
    align: str = "center",
    spacing: int = 4,
) -> np.ndarray:
    face = ImageFont.truetype(FONTS.get(font, font), size)
    style = {"font": face, "align": align, "spacing": spacing, "stroke_width": stroke}
    box = ImageDraw.Draw(Image.new("RGBA", (1, 1))).multiline_textbbox((0, 0), text, **style)
    left, top, right, bottom = math.floor(box[0]), math.floor(box[1]), math.ceil(box[2]), math.ceil(box[3])
    pad = stroke + 4
    image = Image.new("RGBA", (right - left + 2 * pad, bottom - top + 2 * pad))
    ImageDraw.Draw(image).multiline_text((pad - left, pad - top), text, fill=color[::-1], stroke_fill=stroke_color[::-1], **style)
    return premultiply(cv2.cvtColor(np.array(image), cv2.COLOR_RGBA2BGRA))


def row(*sprites: np.ndarray, gap: int = 0) -> np.ndarray:
    height = max(s.shape[0] for s in sprites)
    width = sum(s.shape[1] for s in sprites) + gap * (len(sprites) - 1)
    out = np.zeros((height, width, 4), np.uint8)
    x = 0
    for s in sprites:
        y = (height - s.shape[0]) // 2
        out[y : y + s.shape[0], x : x + s.shape[1]] = s
        x += s.shape[1] + gap
    return out


def wrap_text(text: str, font: str, size: int, width: int) -> list[str]:
    face = ImageFont.truetype(FONTS.get(font, font), size)
    lines, line = [], ""
    for word in text.split():
        candidate = f"{line} {word}".strip()
        if line and face.getlength(candidate) > width:
            lines.append(line)
            line = word
        else:
            line = candidate
    return lines + [line]
