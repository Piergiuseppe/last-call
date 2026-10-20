import shutil
import sys
import time
import unicodedata
from collections.abc import Sequence
from collections.abc import Set as AbstractSet
from typing import Self

from ..domain.gesture import Gesture
from .catalog import Entry

GREEN, YELLOW, DIM, BOLD, RESET = "\x1b[32m", "\x1b[33m", "\x1b[2m", "\x1b[1m", "\x1b[0m"
HIDE_CURSOR, SHOW_CURSOR, CLEAR_LINE = "\x1b[?25l", "\x1b[?25h", "\x1b[2K"
LEGEND = "▶ attivo   … gesto visto, tienilo ancora un attimo   Ctrl+C per uscire"
KEYS_LEGEND = "   (i tasti valgono nella finestra di anteprima)"


def header(camera: str, virtual_camera: str | None) -> list[str]:
    if virtual_camera is None:
        return [f"last-call   {camera}", "Solo anteprima, niente webcam virtuale."]
    return [f"last-call   {camera} → {virtual_camera}", f"Nell'app di videochiamata scegli «{virtual_camera}»."]


def _columns(text: str) -> int:
    return sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in text)


def _fit_line(text: str, columns: int) -> str:
    if _columns(text) <= columns:
        return text
    while _columns(text) > columns - 1:
        text = text[:-1]
    return text + "…"


def _move_to_top(lines: int) -> str:
    return f"\x1b[{lines}F"


class Board:
    REFRESH = 0.2
    LINGER = 2.0

    def __init__(self, header: list[str], entries: Sequence[Entry], show_keys: bool) -> None:
        self.header = header
        self.entries = entries
        self.show_keys = show_keys
        self.live = sys.stdout.isatty()
        self.drawn = 0
        self.last = -1.0
        self.lit: dict[Gesture, float] = {}

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_) -> None:
        if self.live:
            sys.stdout.write(SHOW_CURSOR)
            sys.stdout.flush()

    def show(self, gesture: Gesture | None = None, active: AbstractSet[Gesture] = frozenset(), fps: float | None = None) -> None:
        now = time.monotonic()
        self.lit.update(dict.fromkeys(active, now))
        if self.drawn and (not self.live or now - self.last < self.REFRESH):
            return
        self.last = now
        lit = {g for g, at in self.lit.items() if now - at < self.LINGER}
        lines = self._lines(gesture, lit, fps)
        out = [HIDE_CURSOR] if self.live else []
        if self.drawn:
            out.append(_move_to_top(self.drawn))
        out += [f"{CLEAR_LINE}{line}\n" for line in lines]
        sys.stdout.write("".join(out))
        sys.stdout.flush()
        self.drawn = len(lines)

    def _lines(self, gesture: Gesture | None, lit: AbstractSet[Gesture], fps: float | None) -> list[str]:
        columns = shutil.get_terminal_size().columns - 1
        hint_width = max(len(e.hint) for e in self.entries) + 2
        lines = [BOLD + _fit_line(self.header[0], columns) + RESET]
        lines += [_fit_line(line, columns) for line in self.header[1:]]
        lines.append(DIM + (f"{fps:.0f} fps" if fps else "avvio...") + RESET)
        lines.append("")
        for entry in self.entries:
            key = f"[{entry.key}] " if self.show_keys else ""
            if entry.gesture in lit:
                mark, color = "▶", GREEN + BOLD
            elif entry.gesture == gesture:
                mark, color = "…", YELLOW
            else:
                mark, color = " ", ""
            lines.append(color + _fit_line(f" {mark} {key}{entry.hint:<{hint_width}}{entry.title}", columns) + RESET)
        lines.append("")
        legend = LEGEND + (KEYS_LEGEND if self.show_keys else "")
        lines.append(DIM + _fit_line(legend, columns) + RESET)
        return lines
