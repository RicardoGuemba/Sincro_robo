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
    stable: bool = False,
) -> np.ndarray:
    annotated = frame.copy()
    if mask is not None:
        binary = np.asarray(mask, dtype=bool)
        overlay = annotated.copy()
        overlay[binary] = (50, 209, 176)
        annotated = cv2.addWeighted(overlay, 0.38, annotated, 0.62, 0)
        contours, _hierarchy = cv2.findContours(
            binary.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(annotated, contours, -1, (115, 255, 226), 2, cv2.LINE_AA)
    if x is not None and y is not None and angle_deg is not None:
        center = (int(round(x)), int(round(y)))
        length = 125
        radians_value = np.radians(angle_deg)
        direction = np.asarray([np.cos(radians_value), np.sin(radians_value)])
        start = tuple(np.rint(np.asarray(center) - direction * length).astype(int))
        end = tuple(np.rint(np.asarray(center) + direction * length).astype(int))
        color = (71, 231, 186) if stable else (40, 183, 255)
        cv2.line(annotated, start, end, color, 4, cv2.LINE_AA)
        cv2.circle(annotated, center, 8, (245, 250, 252), 2, cv2.LINE_AA)
        label = f"theta={angle_deg:05.1f} deg"
        cv2.putText(
            annotated,
            label,
            (center[0] + 14, max(28, center[1] - 14)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.66,
            color,
            2,
            cv2.LINE_AA,
        )
    return annotated

