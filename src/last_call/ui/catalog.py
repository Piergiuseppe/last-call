from dataclasses import dataclass

from ..domain.gesture import Gesture


@dataclass(frozen=True)
class Entry:
    gesture: Gesture
    hint: str
    title: str
    key: str


CATALOG = (
    Entry(Gesture.SQUIRTLE, "indici e pollici che si toccano, a ovale", "Squirtle 💦", "w"),
    Entry(Gesture.MIDDLE_FINGER, "dito medio (anche con due mani)", "pioggia di 🖕", "m"),
    Entry(Gesture.EXORCISM, "indici incrociati a croce", "Esorcismo", "e"),
    Entry(Gesture.HAND_OF_GOD, "indice dritto in su, una mano sola", "Mano di Dio", "d"),
    Entry(Gesture.FUNERAL, "braccia incrociate, mani sulle spalle opposte", "Manifesto funebre ⚰️", "f"),
    Entry(Gesture.WASTED, "pistola con le dita, di lato, una mano sola", "WASTED", "g"),
    Entry(Gesture.HEART_MONITOR, "indice e medio uniti sul collo", "Elettrocardiogramma", "c"),
    Entry(Gesture.BEATIFICATION, "mani giunte", "Beatificazione, Dio 🐶", "b"),
    Entry(Gesture.REAPER, "segno del time-out, mani a T", "La Morte guarda l'orologio ⏳", "o"),
    Entry(Gesture.ELECTRIC_CHAIR, "mani nei capelli, tutte e due sulla testa", "Sedia elettrica ⚡", "s"),
    Entry(Gesture.CREMATION, "polsi uniti o incrociati, dita aperte in su", "Cremazione ⚱️", "r"),
    Entry(Gesture.HANDS_UP, "mani in alto aperte, ai lati della testa", "Cecchini 🎯", "a"),
    Entry(Gesture.TESTAMENT, "mano destra aperta sul cuore", "Testamento 📜", "t"),
    Entry(Gesture.GRAVE, "nessuno in camera per 3 secondi", "Lapide, poi zombie 🪦", "l"),
)
