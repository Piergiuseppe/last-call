import numpy as np

from ..domain.scene import Scene

DEFAULT_NAME = "il caro estinto"
DESIGN_HEIGHT = 720


class Effect:
    def __init__(self, width: int, height: int, name: str = DEFAULT_NAME, seed: int | None = None) -> None:
        self.w, self.h = width, height
        self.name = name
        self.unit = height / DESIGN_HEIGHT
        self.rng = np.random.default_rng(seed)
        self.setup()

    def setup(self) -> None:
        pass

    def render(self, frame: np.ndarray, scene: Scene, on: bool, t: float, dt: float) -> np.ndarray:
        raise NotImplementedError


class OneShot(Effect):
    DURATION = 5.0

    def __init__(self, width: int, height: int, name: str = DEFAULT_NAME, seed: int | None = None) -> None:
        self.start: float | None = None
        self.was_on = False
        super().__init__(width, height, name, seed)

    def render(self, frame: np.ndarray, scene: Scene, on: bool, t: float, dt: float) -> np.ndarray:
        if on and not self.was_on and self.start is None:
            self.start = t
            self.begin(scene)
        self.was_on = on
        if self.start is not None and t - self.start >= self.DURATION:
            self.start = None
        if self.start is None:
            return frame
        return self.draw(frame, scene, t - self.start, dt)

    def begin(self, scene: Scene) -> None:
        pass

    def draw(self, frame: np.ndarray, scene: Scene, elapsed: float, dt: float) -> np.ndarray:
        raise NotImplementedError
