class Hold:
    def __init__(self, hold: float, grace: float) -> None:
        self.hold, self.grace = hold, grace
        self.since: float | None = None
        self.last_seen = 0.0

    def update(self, seen: bool, t: float) -> bool:
        if seen:
            if self.since is None:
                self.since = t
            self.last_seen = t
        elif self.since is not None and t - self.last_seen >= self.grace:
            self.since = None
        return self.since is not None and t - self.since >= self.hold
