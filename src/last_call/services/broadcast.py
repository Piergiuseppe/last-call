from typing import Protocol, Self

import numpy as np
import pyvirtualcam

UNAVAILABLE = (
    "Webcam virtuale di OBS non disponibile ({error}).\n"
    'Attiva "OBS Virtual Camera" in Impostazioni di Sistema > Generali > Elementi di login ed estensioni > '
    'Estensioni fotocamera, poi in OBS premi una volta "Avvia fotocamera virtuale" e fermala.\n'
    "Per provare solo in anteprima: last-call --preview --no-virtualcam"
)


class VirtualCameraError(Exception):
    pass


class Broadcast(Protocol):
    @property
    def device(self) -> str | None: ...

    def send(self, frame: np.ndarray) -> None: ...


class Offline:
    device = None

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_) -> None:
        pass

    def send(self, frame: np.ndarray) -> None:
        pass


class BroadcastService:
    def __init__(self, width: int, height: int, fps: int) -> None:
        self.width, self.height, self.fps = width, height, fps

    def open(self, enabled: bool) -> pyvirtualcam.Camera | Offline:
        if not enabled:
            return Offline()
        try:
            return pyvirtualcam.Camera(self.width, self.height, self.fps, fmt=pyvirtualcam.PixelFormat.BGR, backend="obs")
        except RuntimeError as error:
            raise VirtualCameraError(UNAVAILABLE.format(error=error)) from error
