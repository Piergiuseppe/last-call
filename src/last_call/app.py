import time
from contextlib import ExitStack, suppress
from dataclasses import dataclass

import numpy as np

from .services.broadcast import Broadcast, BroadcastService
from .services.camera import Camera, CameraService
from .services.effects import EffectService
from .services.gestures import GestureService
from .services.perception import PerceptionService
from .ui.board import Board, header
from .ui.catalog import CATALOG
from .ui.preview import Preview, PreviewClosed

WIDTH, HEIGHT, FPS = 1280, 720, 30
LONGEST_STEP = 0.1


@dataclass(frozen=True)
class Settings:
    camera: str | None
    name: str
    virtual_camera: bool
    preview: bool
    debug: bool
    mirror: bool


class Clock:
    def __init__(self) -> None:
        self.start = self.last = time.monotonic()
        self.fps = 0.0

    def tick(self) -> tuple[float, float]:
        now = time.monotonic()
        step = now - self.last
        self.fps = 0.9 * self.fps + 0.1 / max(step, 1e-3)
        self.last = now
        return now - self.start, min(step, LONGEST_STEP)


@dataclass
class Session:
    camera: Camera
    perception: PerceptionService
    gestures: GestureService
    effects: EffectService
    broadcast: Broadcast
    board: Board
    preview: Preview | None

    def run(self) -> None:
        self.board.show()
        clock = Clock()
        while True:
            frame = self.camera.read()
            if frame is not None:
                self._process(frame, *clock.tick(), clock.fps)

    def _process(self, frame: np.ndarray, t: float, dt: float, fps: float) -> None:
        scene = self.perception.perceive(frame, t)
        recognition = self.gestures.update(scene, t)
        active = self.effects.active(recognition.active, t)
        out = self.effects.render(frame, scene, active, t, dt)
        self.broadcast.send(out)
        self.board.show(recognition.gesture, active, fps)
        if self.preview is not None:
            self.preview.show(out, scene, recognition.gesture, active, fps)
            if (requested := self.preview.requested()) is not None:
                self.effects.force(requested, t)


def run(settings: Settings) -> None:
    with ExitStack() as stack:
        camera = stack.enter_context(CameraService(WIDTH, HEIGHT).open(settings.camera))
        perception = stack.enter_context(PerceptionService(WIDTH, HEIGHT))
        effects = EffectService(WIDTH, HEIGHT, settings.name)
        broadcast = stack.enter_context(BroadcastService(WIDTH, HEIGHT, FPS).open(settings.virtual_camera))
        board = stack.enter_context(Board(header(camera.name, broadcast.device), CATALOG, show_keys=settings.preview))
        preview = stack.enter_context(Preview(CATALOG, settings.debug, settings.mirror)) if settings.preview else None
        with suppress(KeyboardInterrupt, PreviewClosed):
            Session(camera, perception, GestureService(), effects, broadcast, board, preview).run()
