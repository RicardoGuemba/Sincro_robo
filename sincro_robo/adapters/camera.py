from __future__ import annotations

import threading
import time
from typing import Any

import cv2
import numpy as np


DEFAULT_MAX_DIMENSION = 960


def limit_frame_resolution(
    frame: np.ndarray,
    max_dimension: int = DEFAULT_MAX_DIMENSION,
) -> np.ndarray:
    """Downscale so the longer side is at most max_dimension, keeping aspect.

    A 4:3 Sentech frame (2592×1944) becomes 960×720, the v2108 pick output.
    Frames already within the cap are returned unchanged. Never upscales.
    """
    limit = int(max_dimension)
    if limit <= 0 or frame.size == 0:
        return frame
    height, width = int(frame.shape[0]), int(frame.shape[1])
    if width <= limit and height <= limit:
        return frame
    scale = min(limit / width, limit / height)
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    return cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)


class SyntheticCamera:
    """Repeatable camera simulator used only for local development and tests."""

    def __init__(self, width: int = 960, height: int = 720) -> None:
        self.width = int(width)
        self.height = int(height)
        self._opened = False
        self._started_at = 0.0

    def open(self) -> None:
        self._opened = True
        self._started_at = time.monotonic()

    def grab(self) -> np.ndarray:
        if not self._opened:
            raise RuntimeError("Câmera simulada não está aberta")
        elapsed = time.monotonic() - self._started_at
        positions = (
            (0.50, 0.50, 0.0),
            (0.24, 0.25, 28.0),
            (0.76, 0.25, 62.0),
            (0.24, 0.75, 118.0),
            (0.76, 0.75, 151.0),
            (0.16, 0.50, 87.0),
            (0.84, 0.50, 174.0),
            (0.50, 0.18, 43.0),
            (0.50, 0.82, 136.0),
        )
        px, py, angle = positions[int(elapsed // 10.0) % len(positions)]
        jitter_x = 0.35 * np.sin(elapsed * 2.7)
        jitter_y = 0.35 * np.cos(elapsed * 2.3)
        jitter_angle = 0.18 * np.sin(elapsed * 1.9)
        center = (int(px * self.width + jitter_x), int(py * self.height + jitter_y))

        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        frame[:] = (17, 25, 34)
        for x in range(0, self.width, 80):
            cv2.line(frame, (x, 0), (x, self.height), (31, 43, 55), 1)
        for y in range(0, self.height, 80):
            cv2.line(frame, (0, y), (self.width, y), (31, 43, 55), 1)
        rect = (center, (260, 132), float(angle + jitter_angle))
        box = cv2.boxPoints(rect).astype(np.int32)
        cv2.fillConvexPoly(frame, box, (56, 205, 185))
        cv2.polylines(frame, [box], True, (178, 255, 240), 3, cv2.LINE_AA)
        cv2.circle(frame, center, 10, (12, 22, 30), -1, cv2.LINE_AA)
        cv2.putText(
            frame,
            "SIMULADOR DE CAMPO",
            (24, 38),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (150, 166, 180),
            2,
            cv2.LINE_AA,
        )
        return frame

    def close(self) -> None:
        self._opened = False


class StApiCamera:
    """Omron Sentech adapter; its full lifecycle stays on one owner thread."""

    _BAYER_NAMES = {
        "BayerRG": cv2.COLOR_BayerRGGB2BGR,
        "BayerGR": cv2.COLOR_BayerGRBG2BGR,
        "BayerGB": cv2.COLOR_BayerGBRG2BGR,
        "BayerBG": cv2.COLOR_BayerBGGR2BGR,
    }

    def __init__(
        self,
        device_index: int = 0,
        fetch_timeout_ms: int = 400,
        max_dimension: int = DEFAULT_MAX_DIMENSION,
    ) -> None:
        self.device_index = int(device_index)
        self.fetch_timeout_ms = int(fetch_timeout_ms)
        self.max_dimension = int(max_dimension)
        self._st: Any = None
        self._device: Any = None
        self._datastream: Any = None
        self._initialized = False
        self._owner_thread: int | None = None

    def _assert_owner(self) -> None:
        if self._owner_thread is not None and threading.get_ident() != self._owner_thread:
            raise RuntimeError("O ciclo StApi deve permanecer na thread proprietária")

    def open(self) -> None:
        if self._owner_thread is not None:
            raise RuntimeError("Câmera StApi já está aberta")
        self._owner_thread = threading.get_ident()
        try:
            from ..config import apply_sentech_environment

            apply_sentech_environment()
            import stapipy as st  # type: ignore[import-not-found]

            self._st = st
            st.initialize()
            self._initialized = True
            system = st.create_system()
            if self.device_index <= 0:
                self._device = system.create_first_device()
            else:
                interface = system.create_first_interface()
                self._device = interface.create_device_by_index(self.device_index)
            self._datastream = self._device.create_datastream()
            self._datastream.start_acquisition()
            self._device.acquisition_start()
        except Exception:
            self.close()
            raise

    def _bayer_code(self, pixel_format: Any, info: Any) -> int:
        color_filter = None
        if info is not None and hasattr(info, "get_pixel_color_filter"):
            try:
                color_filter = info.get_pixel_color_filter()
            except Exception:
                color_filter = None
        enum = getattr(self._st, "EStPixelColorFilter", None)
        if color_filter is not None and enum is not None:
            for token, code in self._BAYER_NAMES.items():
                enum_value = getattr(enum, token, None)
                if enum_value is not None and color_filter == enum_value:
                    return code
        text = str(color_filter or pixel_format or "").lower()
        for token, code in self._BAYER_NAMES.items():
            if token.lower() in text:
                return code
        return cv2.COLOR_BayerGRBG2BGR

    def _to_bgr(self, image: Any) -> np.ndarray:
        data = image.get_image_data()
        width, height = int(image.width), int(image.height)
        pixel_format = getattr(image, "pixel_format", "")
        try:
            info = self._st.get_pixel_format_info(pixel_format)
        except Exception:
            info = None
        bits = int(getattr(info, "each_component_total_bit_count", 8) or 8) if info else 8
        is_bayer = bool(getattr(info, "is_bayer", False)) if info else "bayer" in str(pixel_format).lower()
        is_mono = bool(getattr(info, "is_mono", False)) if info else "mono" in str(pixel_format).lower()
        if bits > 8:
            pixels = np.frombuffer(data, np.uint16).copy()
            valid_bits = int(getattr(info, "each_component_valid_bit_count", bits) or bits) if info else bits
            divisor = 2 ** max(0, valid_bits - 8)
            pixels = (pixels / divisor).astype(np.uint8) if divisor else pixels.astype(np.uint8)
        else:
            pixels = np.frombuffer(data, np.uint8).copy()
        pixels = pixels.reshape(height, width, 1)
        if is_bayer:
            pixels = cv2.cvtColor(pixels, self._bayer_code(pixel_format, info))
        elif is_mono or pixels.shape[2] == 1:
            pixels = cv2.cvtColor(pixels, cv2.COLOR_GRAY2BGR)
        return np.ascontiguousarray(pixels)

    def _read_buffer(self, buffer: Any) -> np.ndarray:
        info = getattr(buffer, "info", None)
        if info is not None and not bool(getattr(info, "is_image_present", True)):
            raise TimeoutError("Buffer StApi sem imagem")
        return self._to_bgr(buffer.get_image())

    def grab(self) -> np.ndarray:
        self._assert_owner()
        if self._datastream is None:
            raise RuntimeError("Câmera StApi não está aberta")
        retrieved = self._datastream.retrieve_buffer(self.fetch_timeout_ms)
        if retrieved is None:
            raise TimeoutError("Timeout ao aguardar frame StApi")
        if hasattr(retrieved, "__enter__"):
            with retrieved as buffer:
                return limit_frame_resolution(self._read_buffer(buffer), self.max_dimension)
        try:
            return limit_frame_resolution(self._read_buffer(retrieved), self.max_dimension)
        finally:
            release = getattr(retrieved, "release", None)
            if callable(release):
                release()

    def close(self) -> None:
        self._assert_owner()
        try:
            if self._device is not None:
                try:
                    self._device.acquisition_stop()
                except Exception:
                    pass
            if self._datastream is not None:
                try:
                    self._datastream.stop_acquisition()
                except Exception:
                    pass
            self._device = None
            self._datastream = None
            if self._initialized and self._st is not None:
                self._st.terminate()
        finally:
            self._initialized = False
            self._st = None
            self._owner_thread = None

