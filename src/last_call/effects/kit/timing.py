def smoothstep(edge0: float, edge1: float, x: float) -> float:
    k = min(1.0, max(0.0, (x - edge0) / (edge1 - edge0)))
    return k * k * (3 - 2 * k)


def envelope(elapsed: float, duration: float, attack: float, release: float) -> float:
    return smoothstep(0, attack, elapsed) * (1 - smoothstep(duration - release, duration, elapsed))


def ease_toward(current, target, pull: float):
    return target if current is None else (1 - pull) * current + pull * target
