from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image


@dataclass(frozen=True)
class SegmentationResult:
    masks: tuple[np.ndarray, ...]
    confidences: tuple[float, ...]
    class_names: tuple[str, ...]


class SyntheticSegmenter:
    def load(self) -> None:
        return None

    def predict(self, frame_bgr: np.ndarray) -> SegmentationResult:
        # The simulator paints the object in a unique turquoise range.
        blue, green, red = cv2.split(frame_bgr)
        binary = ((green > 150) & (red > 130) & (blue < 120)).astype(np.uint8)
        count, labels, stats, _centroids = cv2.connectedComponentsWithStats(binary, 8)
        masks: list[np.ndarray] = []
        for label in range(1, count):
            if int(stats[label, cv2.CC_STAT_AREA]) >= 200:
                masks.append(labels == label)
        return SegmentationResult(
            masks=tuple(masks),
            confidences=tuple(0.995 for _ in masks),
            class_names=tuple("sku" for _ in masks),
        )


class RFDetrSegmenter:
    def __init__(
        self,
        checkpoint: str,
        threshold: float = 0.3,
        resolution: int = 384,
        class_name: str = "sku",
        device: str = "auto",
    ) -> None:
        self.checkpoint = str(Path(checkpoint).resolve())
        self.threshold = float(threshold)
        self.resolution = int(resolution)
        self.class_name = class_name
        self.device = device
        self._model: Any = None

    def load(self) -> None:
        if not Path(self.checkpoint).is_file():
            raise FileNotFoundError(f"Checkpoint RF-DETR não encontrado: {self.checkpoint}")
        from rfdetr import RFDETRSegSmall  # type: ignore[import-not-found]

        self._model = RFDETRSegSmall(
            pretrain_weights=self.checkpoint,
            resolution=self.resolution,
        )
        dummy = Image.fromarray(
            np.zeros((self.resolution, self.resolution, 3), dtype=np.uint8)
        )
        self._model.predict(dummy, threshold=self.threshold)

    def predict(self, frame_bgr: np.ndarray) -> SegmentationResult:
        if self._model is None:
            raise RuntimeError("Modelo RF-DETR não foi carregado")
        image = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        detections = self._model.predict(image, threshold=self.threshold)
        raw_masks = getattr(detections, "mask", None)
        if raw_masks is None:
            return SegmentationResult((), (), ())
        masks_array = np.asarray(raw_masks, dtype=bool)
        if masks_array.ndim == 2:
            masks_array = masks_array[np.newaxis, ...]
        confidences = getattr(detections, "confidence", None)
        confidence_values = (
            [float(value) for value in confidences]
            if confidences is not None
            else [1.0] * len(masks_array)
        )
        class_names: list[str] = []
        data = getattr(detections, "data", {}) or {}
        names = data.get("class_name") if isinstance(data, dict) else None
        for index in range(len(masks_array)):
            class_names.append(str(names[index]) if names is not None else self.class_name)
        return SegmentationResult(
            masks=tuple(np.asarray(mask, dtype=bool) for mask in masks_array),
            confidences=tuple(confidence_values),
            class_names=tuple(class_names),
        )


def annotate_frame(
    frame: np.ndarray,
    mask: np.ndarray | None,
    x: float | None = None,
    y: float | None = None,
    angle_deg: float | None = None,
    *,
    major_axis_length: float | None = None,
    stroke_scale: float = 1.0,
    minor_axis_length: float | None = None,
    vcpn_x: float | None = None,
    vcpn_y: float | None = None,
    roi_px: tuple[float, float, float, float] | list[float] | None = None,
    roi_enabled: bool = False,
    roi_quadrant: str | None = None,
    confidence: float | None = None,
    mask_area_cm2: float | None = None,
    label: str = "Embalagem",
    overlay: str = "pick",
) -> np.ndarray:
    from ..geometry import parse_roi_px
    from ..overlay import annotate_pick_overlay, annotate_vcpn_overlay

    if overlay == "vcpn" or roi_enabled or vcpn_x is not None:
        parsed_roi = parse_roi_px(roi_px) if roi_px is not None or roi_enabled else None
        return annotate_vcpn_overlay(
            frame,
            mask,
            x,
            y,
            angle_deg,
            vcpn_x=vcpn_x,
            vcpn_y=vcpn_y,
            roi_px=parsed_roi,
            roi_enabled=roi_enabled,
            roi_quadrant=roi_quadrant,
            confidence=confidence,
            mask_area_cm2=mask_area_cm2,
            label=label,
            stroke_scale=stroke_scale,
        )
    return annotate_pick_overlay(
        frame,
        mask,
        x,
        y,
        angle_deg,
        major_axis_length,
        stroke_scale,
        minor_axis_length,
    )

