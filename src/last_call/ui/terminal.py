import os
import sys
from pathlib import Path

NATIVE_LOG = Path.home() / "Library" / "Logs" / "last-call.log"


def redirect_native_logs() -> None:
    terminal = os.dup(2)
    sys.stderr = os.fdopen(terminal, "w", buffering=1)
    NATIVE_LOG.parent.mkdir(parents=True, exist_ok=True)
    log = os.open(NATIVE_LOG, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    os.dup2(log, 2)
    os.close(log)
