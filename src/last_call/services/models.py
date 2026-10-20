import urllib.request
from pathlib import Path

MODELS_DIR = Path(__file__).parent.parent / "models"
MODELS_URL = "https://storage.googleapis.com/mediapipe-models/"
HAND_MODEL = "hand_landmarker.task"
FACE_MODEL = "face_landmarker.task"
POSE_MODEL = "pose_landmarker_lite.task"
SEGMENTATION_MODEL = "selfie_segmenter.tflite"
SOURCES = {
    HAND_MODEL: "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
    FACE_MODEL: "face_landmarker/face_landmarker/float16/latest/face_landmarker.task",
    POSE_MODEL: "pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
    SEGMENTATION_MODEL: "image_segmenter/selfie_segmenter/float16/latest/selfie_segmenter.tflite",
}


def model_path(name: str) -> str:
    path = MODELS_DIR / name
    if not path.exists():
        print(f"Scarico il modello {name}...")
        MODELS_DIR.mkdir(exist_ok=True)
        urllib.request.urlretrieve(MODELS_URL + SOURCES[name], path)
    return str(path)
