import cv2
import numpy as np

from ..kit.compositing import blit
from ..kit.noise import smooth_noise
from ..kit.sprites import emoji_sprite, premultiply, text_sprite

STONE = (124, 128, 126)
MOSS = (52, 98, 70)
CRACK = (64, 66, 64)
SOIL = [(30, 44, 62), (24, 36, 52), (40, 58, 80), (20, 28, 38)]
ENGRAVED, LIT_EDGE = (46, 48, 46), (196, 200, 196)
FOG = (200, 190, 175)
CANDLE_PLASTIC, CANDLE_LID, CANDLE_LID_RIM = (30, 28, 185), (60, 150, 200), (50, 125, 170)
MOONLIT_FLOWERS = np.array([0.7, 0.6, 0.55])


def moonlit_flowers(char: str) -> np.ndarray:
    sprite = emoji_sprite(char).copy()
    sprite[..., :3] = (sprite[..., :3] * MOONLIT_FLOWERS).astype(np.uint8)
    return sprite


def gravestone(rng: np.random.Generator, unit: float, name: str, date: str) -> np.ndarray:
    u = unit
    width, body, base, margin = int(370 * u), int(455 * u), int(42 * u), int(30 * u)
    shape = (margin + body + base, width + 2 * margin + int(40 * u))
    cx, radius = shape[1] // 2, width // 2
    silhouette = _silhouette(shape, cx, radius, margin, body, u)
    stone = _weathered_stone(rng, silhouette, u)
    _crack_from_edge(stone, rng, (cx, margin + radius), radius, u)
    _carve_cross(stone, cx, margin + int(34 * u), u)
    _carve_epitaph(stone, cx, margin, width * 0.84, u, name, date)
    alpha = (silhouette * 255).astype(np.uint8)[..., None]
    return premultiply(np.concatenate([stone, alpha], axis=2))


def _silhouette(shape: tuple[int, int], cx: int, radius: int, margin: int, body: int, u: float) -> np.ndarray:
    mask = np.zeros(shape, np.float32)
    cv2.circle(mask, (cx, margin + radius), radius, 1.0, -1, cv2.LINE_AA)
    cv2.rectangle(mask, (cx - radius, margin + radius), (cx + radius, margin + body), 1.0, -1)
    plinth = radius + int(22 * u)
    cv2.rectangle(mask, (cx - plinth, margin + body), (cx + plinth, shape[0]), 1.0, -1)
    return mask


def _weathered_stone(rng: np.random.Generator, silhouette: np.ndarray, u: float) -> np.ndarray:
    shape = silhouette.shape
    height = shape[0]
    stone = np.full((*shape, 3), STONE, np.float32)
    stone += (smooth_noise(rng, shape, int(45 * u)) * 9 + smooth_noise(rng, shape, int(7 * u)) * 6)[..., None]
    stone += rng.normal(0, 4, (*shape, 1)).astype(np.float32)
    stone += _bevel_lit_from_top_left(silhouette, u)[..., None]
    depth = np.arange(height, dtype=np.float32)[:, None, None] / height
    stone *= 1.08 - 0.3 * depth
    edge = 1 - cv2.GaussianBlur(silhouette, (0, 0), 18 * u)
    moss = np.clip(smooth_noise(rng, shape, int(28 * u)) * 0.6 + depth[..., 0] * 1.3 + edge * 1.5 - 1.35, 0, 1)
    stone = stone * (1 - 0.75 * moss[..., None]) + np.array(MOSS, np.float32) * 0.75 * moss[..., None]
    return np.clip(stone, 0, 255).astype(np.uint8)


def _bevel_lit_from_top_left(silhouette: np.ndarray, u: float) -> np.ndarray:
    gy, gx = np.gradient(cv2.GaussianBlur(silhouette, (0, 0), 5 * u))
    bevel = gx + gy
    return bevel / (np.abs(bevel).max() + 1e-6) * 60


def _crack_from_edge(stone: np.ndarray, rng: np.random.Generator, center: tuple[int, int], radius: int, u: float) -> None:
    cx, cy = center
    for angle in (-50, 205):
        a = np.radians(angle + rng.normal(0, 6))
        x, y = cx + np.cos(a) * radius * 0.99, cy + np.sin(a) * radius * 0.99
        points = [(x, y)]
        for _ in range(5):
            x -= np.cos(a) * rng.uniform(5, 12) * u + rng.normal(0, 4 * u)
            y += rng.uniform(6, 14) * u
            points.append((x, y))
        cv2.polylines(stone, [np.array(points, np.int32)], False, CRACK, max(1, int(2 * u)), cv2.LINE_AA)


def _carve_cross(stone: np.ndarray, cx: int, top: int, u: float) -> None:
    bar = max(2, int(7 * u))
    for color, shift in ((LIT_EDGE, 1.5 * u), (ENGRAVED, 0)):
        x, y = cx + shift, top + shift
        cv2.line(stone, (int(x), int(y)), (int(x), int(top + 62 * u + shift)), color, bar, cv2.LINE_AA)
        arm = int(top + 19 * u + shift)
        cv2.line(stone, (int(cx - 20 * u + shift), arm), (int(cx + 20 * u + shift), arm), color, bar, cv2.LINE_AA)


def _carve_epitaph(stone: np.ndarray, cx: int, margin: int, max_width: float, u: float, name: str, date: str) -> None:
    lines = [
        ("QUI GIACE", "serif_bold", 34, 128),
        (name.upper(), "serif_bold", 74, 196),
        ("andato a prendere un caffè", "serif_italic", 31, 262),
        (f"† {date}", "serif", 33, 314),
        ("Mai più tornato.", "serif_italic", 33, 372),
    ]
    for text, font, size, y in lines:
        _carve_text(stone, text, font, int(size * u), (cx, margin + y * u), max_width, u)
    rule_y = int(margin + 228 * u)
    start, end = (int(cx - 70 * u), rule_y), (int(cx + 70 * u), rule_y)
    thickness = max(1, int(2 * u))
    cv2.line(stone, (start[0] + 1, start[1] + 1), (end[0] + 1, end[1] + 1), LIT_EDGE, thickness)
    cv2.line(stone, start, end, ENGRAVED, thickness)


def _carve_text(
    stone: np.ndarray, text: str, font: str, size: int, center: tuple[float, float], max_width: float, u: float
) -> None:
    groove = text_sprite(text, font, size, ENGRAVED)
    lit_edge = text_sprite(text, font, size, LIT_EDGE)
    scale = min(1.0, max_width / groove.shape[1])
    x, y = center
    blit(stone, lit_edge, (x + 1.5 * u, y + 1.5 * u), scale, opacity=0.7)
    blit(stone, groove, (x, y), scale)


def ground_shadow(unit: float) -> np.ndarray:
    u = unit
    w, h = int(560 * u), int(90 * u)
    alpha = np.zeros((h, w), np.float32)
    cv2.ellipse(alpha, (w // 2, h // 2), (int(230 * u), int(22 * u)), 0, 0, 360, 1.0, -1)
    alpha = cv2.GaussianBlur(alpha, (0, 0), 12 * u) * 0.75
    shadow = np.zeros((h, w, 4), np.uint8)
    shadow[..., 3] = (alpha * 255).astype(np.uint8)
    return shadow


def grass_strip(rng: np.random.Generator, unit: float, width: int) -> np.ndarray:
    u = unit
    height = int(120 * u)
    color = np.zeros((height, width, 3), np.uint8)
    alpha = np.zeros((height, width), np.uint8)
    soil_top = height - int(34 * u)
    soil_edge = (soil_top + smooth_noise(rng, (1, width), int(40 * u))[0] * 5 * u).astype(np.int32)
    for x in range(width):
        color[soil_edge[x] :, x] = SOIL[1]
        alpha[soil_edge[x] :, x] = 255
    for _ in range(int(width / u * 1.6)):
        x = rng.uniform(0, width)
        root = height - rng.uniform(0, 30) * u
        length = rng.uniform(25, 95) * u
        lean = rng.normal(0, 0.35)
        tip = (x + lean * length, root - length)
        green = rng.uniform(40, 95)
        blade = np.array([(x - 3 * u, root), (x + 3 * u, root), tip], np.int32)
        cv2.fillPoly(color, [blade], (int(green * 0.62), int(green * 0.9), int(green * 0.5)), cv2.LINE_AA)
        cv2.fillPoly(alpha, [blade], 255, cv2.LINE_AA)
    return premultiply(np.dstack([color, alpha]))


def scrolling_fog(rng: np.random.Generator, unit: float, width: int) -> np.ndarray:
    height = int(260 * unit)
    noise = smooth_noise(rng, (height, width), int(120 * unit))
    seamless = np.hstack([noise, noise])
    thicker_near_ground = np.linspace(0, 1, height, dtype=np.float32)[:, None] ** 1.6
    density = np.clip(0.55 + 0.35 * seamless, 0, 1) * thicker_near_ground
    return (density[..., None] * np.array(FOG, np.float32)).astype(np.uint8)


def grave_candle(unit: float) -> np.ndarray:
    u = unit
    w, h = int(62 * u), int(108 * u)
    color = np.zeros((h, w, 3), np.float32)
    alpha = np.zeros((h, w), np.float32)
    lid = int(16 * u)
    jar = np.array([(4 * u, h - 1), (w - 4 * u, h - 1), (w - 8 * u, lid), (8 * u, lid)], np.int32)
    cv2.fillPoly(alpha, [jar], 0.93, cv2.LINE_AA)
    round_jar = 0.55 + 0.45 * np.cos(np.linspace(-1, 1, w, dtype=np.float32) * 1.4)
    lit_from_inside = 1.25 - 0.5 * np.linspace(0, 1, h, dtype=np.float32)[:, None]
    color[:] = np.array(CANDLE_PLASTIC, np.float32) * (round_jar * lit_from_inside)[..., None]
    for y in range(lid + int(14 * u), h - int(6 * u), max(4, int(13 * u))):
        color[y : y + max(1, int(2 * u))] *= 0.7
    color[:, int(w * 0.3) : int(w * 0.3) + max(2, int(4 * u))] += (40, 40, 90)
    cap = ((w // 2, lid), (w // 2 - int(6 * u), int(9 * u)))
    rim = ((int(7 * u), lid), (w - int(7 * u), lid + int(6 * u)))
    cv2.ellipse(color, *cap, 0, 180, 360, CANDLE_LID, -1, cv2.LINE_AA)
    cv2.rectangle(color, *rim, CANDLE_LID_RIM, -1)
    cv2.ellipse(alpha, *cap, 0, 180, 360, 1.0, -1, cv2.LINE_AA)
    cv2.rectangle(alpha, *rim, 1.0, -1)
    return premultiply(np.dstack([np.clip(color, 0, 255).astype(np.uint8), (alpha * 255).astype(np.uint8)]))
