import cv2
import numpy as np

from ..domain.landmarks import INDEX, THUMB, BodyPoint, Finger, HandPoint
from ..domain.scene import Body, Face, Hand, Oval, Scene, Side
from .anatomy import (
    SHOULDERS_SEEN,
    Torso,
    curled,
    distance,
    extended,
    head_offset,
    index_tilt,
    lies_flat,
    mean_size,
    on_face,
    only_index_extended,
    open_hand,
    points_up,
    segments_cross,
    torso_of,
)

EMPTY_ROOM_COVERAGE = 0.05
CROSSING_COSINE = 0.7
ARCHED_INDEX_REACH = 1.05
OVAL_HOLE_SHRINK = 0.8
FLAT_HAND_CONTACTS = (
    HandPoint.WRIST,
    HandPoint.INDEX_MCP,
    HandPoint.INDEX_PIP,
    HandPoint.MIDDLE_MCP,
    HandPoint.MIDDLE_PIP,
    HandPoint.RING_MCP,
    HandPoint.RING_PIP,
    HandPoint.PINKY_MCP,
    HandPoint.PINKY_PIP,
)


def nobody_in_frame(scene: Scene) -> bool:
    return scene.face is None and float(scene.person_mask.mean()) < EMPTY_ROOM_COVERAGE


def squirtle_oval(hands: list[Hand]) -> Oval | None:
    if len(hands) != 2:
        return None
    a, b = hands
    size = mean_size(a, b)
    if not _fingertips_touch(a, b, size) or not all(h.reach(Finger.INDEX) > ARCHED_INDEX_REACH for h in hands):
        return None
    top = (a.tip(Finger.INDEX) + b.tip(Finger.INDEX)) / 2
    bottom = (a.thumb_tip + b.thumb_tip) / 2
    if distance(top, bottom) < 0.7 * size or distance(a.palm, b.palm) < 0.9 * size:
        return None
    if not _on_opposite_sides_of_hole(a, b, top - bottom):
        return None
    return _fit_oval(a, b)


def _fingertips_touch(a: Hand, b: Hand, size: float) -> bool:
    indexes_touch = distance(a.tip(Finger.INDEX), b.tip(Finger.INDEX)) <= 0.4 * size
    return indexes_touch and distance(a.thumb_tip, b.thumb_tip) <= 0.4 * size


def _on_opposite_sides_of_hole(a: Hand, b: Hand, along: np.ndarray) -> bool:
    across = b.palm - a.palm
    return abs(across @ along) <= 0.5 * np.linalg.norm(across) * np.linalg.norm(along)


def _fit_oval(a: Hand, b: Hand) -> Oval:
    rim = np.vstack([a.points[THUMB], a.points[INDEX], b.points[THUMB], b.points[INDEX]]).astype(np.float32)
    (cx, cy), (w, h), angle = cv2.fitEllipse(rim)
    if w < h:
        w, h, angle = h, w, angle + 90
    return Oval(np.array([cx, cy]), np.array([w, h]) / 2 * OVAL_HOLE_SHRINK, angle % 180)


def oval_between_hands(scene: Scene) -> bool:
    return squirtle_oval(scene.hands) is not None


def praying(scene: Scene) -> bool:
    hands = scene.hands
    if len(hands) == 1:
        return _single_praying_hand(hands[0], scene.face, scene.body)
    return len(hands) == 2 and _palms_pressed(*hands)


def _palms_pressed(a: Hand, b: Hand) -> bool:
    size = mean_size(a, b)
    return (
        all(extended(h, Finger.INDEX) and extended(h, Finger.MIDDLE) for h in (a, b))
        and distance(a.tip(Finger.INDEX), b.tip(Finger.INDEX)) < 0.5 * size
        and distance(a.tip(Finger.MIDDLE), b.tip(Finger.MIDDLE)) < 0.5 * size
        and distance(a.palm, b.palm) < 0.6 * size
    )


def _single_praying_hand(hand: Hand, face: Face | None, body: Body | None) -> bool:
    upright = open_hand(hand) and points_up(hand, 30)
    return upright and (_edge_on_below_chin(hand, face) or _wrists_joined(hand, body))


def _edge_on_below_chin(hand: Hand, face: Face | None) -> bool:
    if face is None:
        return False
    side, up = head_offset(face, hand.palm)
    knuckles = distance(hand.points[HandPoint.INDEX_MCP], hand.points[HandPoint.PINKY_MCP])
    return knuckles < 0.35 * hand.size and abs(side) < 0.4 and -1.8 < up < -0.3


def _wrists_joined(hand: Hand, body: Body | None) -> bool:
    if body is None or body.wrists_confidence < 0.5:
        return False
    left, right = body.wrists
    farthest = max(distance(left, hand.wrist), distance(right, hand.wrist))
    return distance(left, right) < 1.5 * hand.size and farthest < 1.5 * hand.size


def flame(scene: Scene) -> bool:
    hands = scene.hands
    if len(hands) != 2 or not all(open_hand(h) and points_up(h, 60) for h in hands):
        return False
    a, b = hands
    size = mean_size(a, b)
    return distance(a.wrist, b.wrist) < 1.3 * size and distance(a.tip(Finger.MIDDLE), b.tip(Finger.MIDDLE)) > size


def crossed_index_fingers(scene: Scene) -> bool:
    hands = scene.hands
    if len(hands) != 2 or not all(extended(h, Finger.INDEX) and not extended(h, Finger.MIDDLE) for h in hands):
        return False
    (a1, a2), (b1, b2) = (h.points[[HandPoint.INDEX_MCP, HandPoint.INDEX_TIP]] for h in hands)
    va, vb = a2 - a1, b2 - b1
    cosine = abs(va @ vb) / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-6)
    return cosine < CROSSING_COSINE and segments_cross(a1, a2, b1, b2)


def hands_on_head(scene: Scene) -> bool:
    face = scene.face
    if len(scene.hands) != 2 or face is None:
        return False
    return all(_on_top_of_head(face, h) for h in scene.hands)


def _on_top_of_head(face: Face, hand: Hand) -> bool:
    side, up = head_offset(face, hand.palm)
    return abs(side) < 0.75 and 0.45 < up < 1.6


def surrender(scene: Scene) -> bool:
    hands, face = scene.hands, scene.face
    if len(hands) != 2 or face is None or not all(open_hand(h) and points_up(h, 40) for h in hands):
        return False
    offsets = [head_offset(face, h.palm) for h in hands]
    (side_a, _), (side_b, _) = offsets
    return all(_beside_head(side, up) for side, up in offsets) and side_a * side_b < 0


def _beside_head(side: float, up: float) -> bool:
    return 0.9 < abs(side) < 4.0 and -0.6 < up < 2.5


def time_out(scene: Scene) -> bool:
    hands = scene.hands
    return len(hands) == 2 and any(_forms_a_t(stem, bar) for stem, bar in (hands, hands[::-1]))


def _forms_a_t(stem: Hand, bar: Hand) -> bool:
    straight = all(extended(h, f) for h in (stem, bar) for f in (Finger.INDEX, Finger.MIDDLE))
    if not (straight and points_up(stem, 35) and lies_flat(bar, 35)):
        return False
    touch = min(distance(stem.tip(Finger.MIDDLE), bar.points[p]) for p in FLAT_HAND_CONTACTS)
    return touch < 0.5 * mean_size(stem, bar) and bar.palm[1] < stem.palm[1]


def hand_on_heart(scene: Scene) -> bool:
    torso = torso_of(scene.face, scene.body)
    if torso is None:
        return False
    return any(_swearing(hand, scene, torso) for hand in scene.hands)


def _swearing(hand: Hand, scene: Scene, torso: Torso) -> bool:
    if hand.side != Side.RIGHT or not all(extended(hand, f) for f in (Finger.INDEX, Finger.MIDDLE, Finger.RING)):
        return False
    if _other_hand_nearby(hand, scene):
        return False
    toward_left = (hand.palm[0] - torso.middle) * torso.left_side
    across = hand.direction[0] * torso.left_side > 0.4 * np.linalg.norm(hand.direction)
    over_heart = -0.1 * torso.width < toward_left < 0.6 * torso.width
    chest_height = torso.chin + 0.3 * torso.width < hand.palm[1] < torso.chin + 1.5 * torso.width
    return across and over_heart and chest_height


def _other_hand_nearby(hand: Hand, scene: Scene) -> bool:
    nearby = [other.palm for other in scene.hands if other is not hand]
    body = scene.body
    if body is not None and body.visibility[BodyPoint.LEFT_WRIST] > 0.5:
        nearby.append(body.points[BodyPoint.LEFT_WRIST])
    reach = 1.5 * hand.size
    return any(min(distance(hand.palm, q), distance(hand.tip(Finger.INDEX), q)) < reach for q in nearby)


def crossed_arms(scene: Scene) -> bool:
    return _hands_on_opposite_shoulders(scene) or _wrists_crossed(scene.body)


def _hands_on_opposite_shoulders(scene: Scene) -> bool:
    hands = scene.hands
    torso = torso_of(scene.face, scene.body)
    if len(hands) != 2 or {h.side for h in hands} != {Side.LEFT, Side.RIGHT} or torso is None:
        return False
    return all(_on_opposite_shoulder(h, torso) for h in hands)


def _on_opposite_shoulder(hand: Hand, torso: Torso) -> bool:
    offset = hand.palm[0] - torso.middle
    own_side = torso.left_side if hand.side == Side.LEFT else -torso.left_side
    if offset * own_side >= 0:
        return False
    inward = hand.direction[0] * np.sign(offset) < -0.3 * np.linalg.norm(hand.direction)
    close_to_body = abs(offset) <= 0.75 * torso.width
    chest_height = torso.chin < hand.palm[1] < torso.chin + 1.2 * torso.width
    return not inward and close_to_body and chest_height


def _wrists_crossed(body: Body | None) -> bool:
    if body is None or body.shoulders_confidence < SHOULDERS_SEEN or body.wrists_confidence < 0.2:
        return False
    left_shoulder, right_shoulder = body.shoulders
    left_wrist, right_wrist = body.wrists
    middle, width = (left_shoulder + right_shoulder) / 2, distance(left_shoulder, right_shoulder)
    toward_right = np.sign(right_shoulder[0] - left_shoulder[0])
    margin = 0.15 * width
    swapped = (left_wrist[0] - middle[0]) * toward_right > margin and (right_wrist[0] - middle[0]) * toward_right < -margin
    on_chest = all(middle[1] - 0.2 * width < w[1] < middle[1] + 1.5 * width for w in (left_wrist, right_wrist))
    return swapped and on_chest


def pulse_check(scene: Scene) -> bool:
    face = scene.face
    if face is None:
        return False
    neck = _neck_area(face)
    return any(_two_fingers_on(neck, hand) for hand in scene.hands)


def _neck_area(face: Face) -> tuple[float, float, float, float]:
    xs = face.oval[:, 0]
    left, right = xs.min(), xs.max()
    margin = 0.35 * (right - left)
    return left - margin, right + margin, face.center[1] + 0.15 * face.height, face.chin[1] + 0.6 * face.height


def _two_fingers_on(area: tuple[float, float, float, float], hand: Hand) -> bool:
    pointing = extended(hand, Finger.INDEX) and extended(hand, Finger.MIDDLE)
    if not (pointing and curled(hand, Finger.RING) and curled(hand, Finger.PINKY)):
        return False
    index, middle = hand.tip(Finger.INDEX), hand.tip(Finger.MIDDLE)
    if distance(index, middle) > 0.45 * hand.size:
        return False
    x, y = (index + middle) / 2
    left, right, top, bottom = area
    return left < x < right and top < y < bottom


def finger_gun(scene: Scene) -> bool:
    if len(scene.hands) != 1:
        return False
    hand = scene.hands[0]
    thumb_up = hand.points[HandPoint.INDEX_MCP][1] - hand.thumb_tip[1] > 0.4 * hand.size
    sideways = 50 < index_tilt(hand) < 130
    return only_index_extended(hand) and thumb_up and sideways and not on_face(scene.face, hand.tip(Finger.INDEX))


def pointing_at_the_sky(scene: Scene) -> bool:
    return len(scene.hands) == 1 and only_index_extended(scene.hands[0]) and index_tilt(scene.hands[0]) < 30


def middle_finger(scene: Scene) -> bool:
    return any(
        extended(h, Finger.MIDDLE) and all(curled(h, f) for f in (Finger.INDEX, Finger.RING, Finger.PINKY)) for h in scene.hands
    )
