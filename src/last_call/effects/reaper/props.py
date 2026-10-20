import math

import cv2
import numpy as np

from ..kit.noise import smooth_noise
from ..kit.sprites import emoji_sprite, premultiply

CLOTH, CLOTH_HIGHLIGHT, HOOD_RIM, HOOD_DARKNESS = (26, 21, 29), (48, 42, 54), (46, 40, 52), (6, 5, 8)
BONE, BONE_EDGE = (200, 214, 224), (40, 42, 52)
BLADE, BLADE_BACK, BLADE_EDGE, FERRULE = (150, 152, 150), (90, 92, 92), (238, 240, 238), (60, 62, 64)
WOOD = [((24, 40, 62), 18, 0), ((44, 74, 110), 12, -1), ((80, 120, 160), 3, -4)]
SKULL = "\U0001f480"
CLOAK_OUTLINE = [
    (20, 800),
    (30, 560),
    (70, 470),
    (135, 425),
    (165, 380),
    (150, 300),
    (160, 215),
    (195, 150),
    (250, 104),
    (312, 88),
    (365, 100),
    (405, 132),
    (430, 180),
    (436, 245),
    (425, 320),
    (438, 390),
    (476, 430),
    (536, 478),
    (574, 570),
    (586, 800),
]


def rounded_corners(points: np.ndarray, rounds: int = 3) -> np.ndarray:
    for _ in range(rounds):
        following = np.roll(points, -1, axis=0)
        points = np.stack([0.75 * points + 0.25 * following, 0.25 * points + 0.75 * following], axis=1).reshape(-1, 2)
    return points


def cloak(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    w, h = 600, 800
    alpha = np.zeros((h, w), np.float32)
    cv2.fillPoly(alpha, [rounded_corners(np.array(CLOAK_OUTLINE, np.float32)).astype(np.int32)], 1.0, cv2.LINE_AA)
    alpha = cv2.GaussianBlur(alpha, (0, 0), 3)
    alpha *= 1 - np.clip((np.arange(h, dtype=np.float32)[:, None] - 0.62 * h) / (0.38 * h), 0, 1) ** 1.5
    cloth = np.full((h, w, 3), CLOTH, np.float32)
    folds = cv2.resize(rng.normal(0, 1, (1, w // 22)).astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
    cloth += (folds * 9 + smooth_noise(rng, (h, w), 10) * 3)[..., None]
    cloth *= (1.3 - 0.5 * np.linspace(0, 1, w, dtype=np.float32))[None, :, None]
    hood = np.array([292.0, 300.0])
    opening = _pointed_arch(hood)
    cv2.polylines(cloth, [opening], True, HOOD_RIM, 12, cv2.LINE_AA)
    cloth = cv2.GaussianBlur(cloth, (0, 0), 1.2)
    darkness = np.zeros((h, w), np.float32)
    cv2.fillPoly(darkness, [opening], 1.0, cv2.LINE_AA)
    darkness = cv2.GaussianBlur(darkness, (0, 0), 5)[..., None]
    cloth = cloth * (1 - darkness) + np.array(HOOD_DARKNESS, np.float32) * darkness
    sprite = premultiply(np.dstack([np.clip(cloth, 0, 255).astype(np.uint8), (np.clip(alpha, 0, 1) * 255).astype(np.uint8)]))
    return sprite, np.array([w / 2, h / 2]) - hood


def _pointed_arch(center: np.ndarray) -> np.ndarray:
    t = np.linspace(0, 2 * np.pi, 120, endpoint=False)
    narrower_on_top = 0.55 + 0.45 * (1 - np.cos(t)) / 2
    return np.stack([center[0] + 88 * np.sin(t) * narrower_on_top, center[1] - 122 * np.cos(t)], axis=1).astype(np.int32)


def skull_in_shadow() -> np.ndarray:
    skull = emoji_sprite(SKULL)
    scale = 128 / skull.shape[0]
    skull = cv2.resize(skull, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    h = skull.shape[0]
    darker_on_top = np.clip(0.12 + 0.75 * np.linspace(0, 1, h, dtype=np.float32), 0, 0.85)[:, None, None]
    shaded = skull.astype(np.float32)
    shaded[..., :3] *= darker_on_top * np.array([1.0, 0.92, 0.84], np.float32)
    sunk_into_hood = 30
    sprite = np.zeros((h + sunk_into_hood, skull.shape[1], 4), np.float32)
    sprite[sunk_into_hood:] = shaded
    return sprite.astype(np.uint8)


def scythe() -> tuple[np.ndarray, np.ndarray]:
    w, h = 420, 680
    color = np.zeros((h, w, 3), np.uint8)
    alpha = np.zeros((h, w), np.uint8)
    x0 = w // 2 + 40
    staff = np.array([(x0 + 5 * math.sin(y / 80), y) for y in range(70, h - 10, 10)], np.int32)
    for shade, thickness, dx in WOOD:
        cv2.polylines(color, [staff + [dx, 0]], False, shade, thickness, cv2.LINE_AA)
    cv2.polylines(alpha, [staff], False, 255, 18, cv2.LINE_AA)
    back, edge = _blade(x0)
    blade = np.vstack([back, edge[::-1]]).astype(np.int32)
    cv2.fillPoly(color, [blade], BLADE, cv2.LINE_AA)
    cv2.fillPoly(alpha, [blade], 255, cv2.LINE_AA)
    cv2.polylines(color, [back.astype(np.int32)], False, BLADE_BACK, 4, cv2.LINE_AA)
    cv2.polylines(color, [edge.astype(np.int32)], False, BLADE_EDGE, 3, cv2.LINE_AA)
    cv2.rectangle(color, (x0 - 14, 62), (x0 + 14, 100), FERRULE, -1)
    cv2.rectangle(alpha, (x0 - 14, 62), (x0 + 14, 100), 255, -1)
    return premultiply(np.dstack([color, alpha])), np.array([x0, 420.0])


def _blade(x0: int) -> tuple[np.ndarray, np.ndarray]:
    along = np.linspace(0, 1, 40)
    spine = np.stack([x0 - 300 * along, 78 + 30 * along + 150 * along**2], axis=1)
    tangent = np.gradient(spine, axis=0)
    normal = np.stack([tangent[:, 1], -tangent[:, 0]], axis=1)
    normal /= np.linalg.norm(normal, axis=1, keepdims=True)
    normal *= -np.sign(normal[0, 1])
    width = 50 * (1 - along) ** 0.8 + 2
    return spine + normal * (width * 0.35)[:, None], spine - normal * (width * 0.65)[:, None]


def bony_hand(pointing: bool) -> np.ndarray:
    w, h = 170, 130
    color = np.zeros((h, w, 3), np.uint8)
    alpha = np.zeros((h, w), np.uint8)
    wrist = np.array([150, 66])
    bones = _hand_bones(wrist, pointing)
    for shade, extra in ((BONE_EDGE, 6), (BONE, 0)):
        for a, b in bones:
            cv2.line(color, tuple(np.int32(a)), tuple(np.int32(b)), shade, 9 + extra, cv2.LINE_AA)
            cv2.line(alpha, tuple(np.int32(a)), tuple(np.int32(b)), 255, 15, cv2.LINE_AA)
        for _, knuckle in bones:
            cv2.circle(color, tuple(np.int32(knuckle)), 7 + extra // 2, shade, -1, cv2.LINE_AA)
            cv2.circle(alpha, tuple(np.int32(knuckle)), 10, 255, -1, cv2.LINE_AA)
    cv2.circle(color, tuple(wrist), 12, BONE_EDGE, -1, cv2.LINE_AA)
    cv2.circle(color, tuple(wrist), 9, BONE, -1, cv2.LINE_AA)
    cv2.circle(alpha, tuple(wrist), 12, 255, -1, cv2.LINE_AA)
    return premultiply(np.dstack([color, alpha]))


def _hand_bones(wrist: np.ndarray, pointing: bool) -> list[tuple[np.ndarray, np.ndarray]]:
    bones = []
    for i in range(4):
        knuckle = np.array([88, 32 + 22 * i])
        bones.append((wrist + [0, (i - 1.5) * 6], knuckle))
        if pointing and i == 0:
            bones += [(knuckle, knuckle + [-34, 2]), (knuckle + [-34, 2], knuckle + [-64, 3])]
        else:
            joint = knuckle + [-26, 4]
            bones += [(knuckle, joint), (joint, joint + [-6, 13 + 2 * i])]
    thumb = np.array([110, 14])
    bones += [(wrist + [-6, -14], thumb), (thumb, thumb + [-22, -4])]
    return bones
