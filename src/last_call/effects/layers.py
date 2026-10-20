from ..domain.gesture import Gesture
from .base import Effect
from .beatification import BeatificationEffect
from .cremation import CremationEffect
from .electric_chair import ElectricChairEffect
from .exorcism import ExorcismEffect
from .funeral import FuneralEffect
from .grave import GraveEffect
from .hand_of_god import HandOfGodEffect
from .hands_up import HandsUpEffect
from .heart_monitor import HeartMonitorEffect
from .middle_finger import MiddleFingerEffect
from .reaper import ReaperEffect
from .squirtle import SquirtleEffect
from .testament import TestamentEffect
from .wasted import WastedEffect

LAYERS: dict[Gesture, type[Effect]] = {
    Gesture.GRAVE: GraveEffect,
    Gesture.CREMATION: CremationEffect,
    Gesture.ELECTRIC_CHAIR: ElectricChairEffect,
    Gesture.EXORCISM: ExorcismEffect,
    Gesture.BEATIFICATION: BeatificationEffect,
    Gesture.SQUIRTLE: SquirtleEffect,
    Gesture.REAPER: ReaperEffect,
    Gesture.HAND_OF_GOD: HandOfGodEffect,
    Gesture.HANDS_UP: HandsUpEffect,
    Gesture.MIDDLE_FINGER: MiddleFingerEffect,
    Gesture.WASTED: WastedEffect,
    Gesture.TESTAMENT: TestamentEffect,
    Gesture.FUNERAL: FuneralEffect,
    Gesture.HEART_MONITOR: HeartMonitorEffect,
}
