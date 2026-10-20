import threading
from typing import Self

import AVFoundation as AVF
import CoreMedia
import cv2
import libdispatch
import numpy as np
import objc
import Quartz
from Foundation import NSObject

ACCESS_DENIED = (
    "Accesso alla fotocamera negato. Consentilo all'app del terminale in "
    "Impostazioni di Sistema > Privacy e sicurezza > Fotocamera, poi riavvia il terminale."
)
SMOOTH_FPS = 29
FIRST_FRAME_TIMEOUT = 2.0


class CameraError(Exception):
    pass


def _name(device) -> str:
    return str(device.localizedName())


def _is_virtual(device) -> bool:
    return "obs" in _name(device).lower()


def _real_devices() -> list:
    session = AVF.AVCaptureDeviceDiscoverySession.discoverySessionWithDeviceTypes_mediaType_position_(
        [AVF.AVCaptureDeviceTypeBuiltInWideAngleCamera, AVF.AVCaptureDeviceTypeExternal],
        AVF.AVMediaTypeVideo,
        AVF.AVCaptureDevicePositionUnspecified,
    )
    return [device for device in session.devices() if not _is_virtual(device)]


def _access_granted() -> bool:
    status = AVF.AVCaptureDevice.authorizationStatusForMediaType_(AVF.AVMediaTypeVideo)
    if status == AVF.AVAuthorizationStatusNotDetermined:
        answered = threading.Event()
        AVF.AVCaptureDevice.requestAccessForMediaType_completionHandler_(AVF.AVMediaTypeVideo, lambda _: answered.set())
        answered.wait()
        status = AVF.AVCaptureDevice.authorizationStatusForMediaType_(AVF.AVMediaTypeVideo)
    return status == AVF.AVAuthorizationStatusAuthorized


def _format(device, width: int, height: int):
    for fmt in device.formats():
        size = CoreMedia.CMVideoFormatDescriptionGetDimensions(fmt.formatDescription())
        smooth = any(r.maxFrameRate() >= SMOOTH_FPS for r in fmt.videoSupportedFrameRateRanges())
        if (size.width, size.height) == (width, height) and smooth:
            return fmt
    return None


def fit(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    h, w = frame.shape[:2]
    target = width / height
    if w / h > target:
        cw = int(h * target)
        frame = frame[:, (w - cw) // 2 : (w - cw) // 2 + cw]
    elif w / h < target:
        ch = int(w / target)
        frame = frame[(h - ch) // 2 : (h - ch) // 2 + ch]
    if frame.shape[:2] != (height, width):
        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    return frame


class _FrameSink(NSObject):
    def init(self):
        sink = objc.super(_FrameSink, self).init()
        sink.ready = threading.Condition()
        sink.frame = None
        sink.count = 0
        return sink

    def captureOutput_didOutputSampleBuffer_fromConnection_(self, output, sample, connection):
        image = CoreMedia.CMSampleBufferGetImageBuffer(sample)
        Quartz.CVPixelBufferLockBaseAddress(image, Quartz.kCVPixelBufferLock_ReadOnly)
        try:
            w, h = Quartz.CVPixelBufferGetWidth(image), Quartz.CVPixelBufferGetHeight(image)
            stride = Quartz.CVPixelBufferGetBytesPerRow(image)
            data = Quartz.CVPixelBufferGetBaseAddress(image).as_buffer(stride * h)
            bgra = np.frombuffer(data, np.uint8).reshape(h, stride)[:, : w * 4].reshape(h, w, 4)
            frame = cv2.cvtColor(bgra, cv2.COLOR_BGRA2BGR)
        finally:
            Quartz.CVPixelBufferUnlockBaseAddress(image, Quartz.kCVPixelBufferLock_ReadOnly)
        with self.ready:
            self.frame = frame
            self.count += 1
            self.ready.notify_all()


class Camera:
    def __init__(self, device, width: int, height: int) -> None:
        self.name = _name(device)
        self.width, self.height = width, height
        self.seen = 0
        self.session = AVF.AVCaptureSession.alloc().init()
        source, error = AVF.AVCaptureDeviceInput.deviceInputWithDevice_error_(device, None)
        if source is None:
            raise CameraError(f"impossibile aprire {self.name}: {error}")
        output = AVF.AVCaptureVideoDataOutput.alloc().init()
        output.setVideoSettings_({Quartz.kCVPixelBufferPixelFormatTypeKey: Quartz.kCVPixelFormatType_32BGRA})
        output.setAlwaysDiscardsLateVideoFrames_(True)
        self.sink = _FrameSink.alloc().init()
        self.queue = libdispatch.dispatch_queue_create(b"last-call.camera", None)
        output.setSampleBufferDelegate_queue_(self.sink, self.queue)
        self.session.addInput_(source)
        self.session.addOutput_(output)
        self.session.startRunning()
        self._capture_at_output_size(device)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_) -> None:
        self.release()

    def _capture_at_output_size(self, device) -> None:
        fmt = _format(device, self.width, self.height)
        if fmt is not None and device.lockForConfiguration_(None)[0]:
            device.setActiveFormat_(fmt)
            device.unlockForConfiguration()

    def read(self, timeout: float = FIRST_FRAME_TIMEOUT) -> np.ndarray | None:
        with self.sink.ready:
            if not self.sink.ready.wait_for(lambda: self.sink.count > self.seen, timeout):
                return None
            self.seen = self.sink.count
            frame = self.sink.frame
        return fit(frame, self.width, self.height)

    def release(self) -> None:
        self.session.stopRunning()


class CameraService:
    def __init__(self, width: int, height: int) -> None:
        self.width, self.height = width, height

    @staticmethod
    def available() -> list[str]:
        return [_name(device) for device in _real_devices()]

    def open(self, wanted: str | None) -> Camera:
        if not _access_granted():
            raise CameraError(ACCESS_DENIED)
        camera = Camera(self._pick(wanted), self.width, self.height)
        if camera.read() is None:
            camera.release()
            raise CameraError(f"Non arrivano immagini da {camera.name}.")
        return camera

    @staticmethod
    def _pick(wanted: str | None):
        devices = _real_devices()
        if wanted is None and devices:
            return devices[0]
        if wanted is not None and wanted.isdigit() and int(wanted) < len(devices):
            return devices[int(wanted)]
        for device in devices:
            if wanted is not None and wanted.lower() in _name(device).lower():
                return device
        names = ", ".join(_name(device) for device in devices) or "nessuna"
        raise CameraError(f"Fotocamera {wanted!r} non trovata. Disponibili: {names}")
