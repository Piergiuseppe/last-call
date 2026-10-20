import argparse
import os
import pwd
import sys

from .app import Settings, run
from .effects.base import DEFAULT_NAME
from .services.broadcast import VirtualCameraError
from .services.camera import CameraError, CameraService
from .ui.catalog import CATALOG
from .ui.terminal import redirect_native_logs


def default_name() -> str:
    full = pwd.getpwuid(os.getuid()).pw_gecos.strip()
    return full.split()[0] if full else DEFAULT_NAME


def parse_args() -> argparse.Namespace:
    keys = ", ".join(f"{entry.key}: {entry.gesture}" for entry in CATALOG)
    parser = argparse.ArgumentParser(description="Webcam virtuale con effetti attivati dai gesti.")
    parser.add_argument("--camera", help="nome (anche parziale) o numero della fotocamera da usare")
    parser.add_argument("--list-cameras", action="store_true", help="elenca le fotocamere ed esce")
    parser.add_argument("--preview", action="store_true", help=f"mostra una finestra di anteprima (tasti {keys}; q esce)")
    parser.add_argument("--debug", action="store_true", help="nell'anteprima disegna mani, volto, spalle e gesto riconosciuto")
    parser.add_argument("--no-virtualcam", action="store_true", help="non inviare alla webcam virtuale di OBS")
    parser.add_argument(
        "--nome", default=default_name(), help="nome su manifesto funebre, lapide e testamento (predefinito: il tuo)"
    )
    parser.add_argument("--specchio", action="store_true", help="anteprima a specchio (le scritte escono al contrario)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.list_cameras:
        for i, name in enumerate(CameraService.available()):
            print(f"{i}: {name}")
        return
    redirect_native_logs()
    settings = Settings(
        camera=args.camera,
        name=args.nome,
        virtual_camera=not args.no_virtualcam,
        preview=args.preview,
        debug=args.debug,
        mirror=args.specchio,
    )
    try:
        run(settings)
    except (CameraError, VirtualCameraError) as error:
        sys.exit(str(error))
