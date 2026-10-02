from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

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
    vcpn_x: float | None = None,
    vcpn_y: float | None = None,
    roi_px: Sequence[float] | None = None,
    roi_enabled: bool = False,
    label: str = "Embalagem",
    confidence: float | None = None,
    mask_area_cm2: float | None = None,
    roi_quadrant: str | None = None,
) -> np.ndarray:
    """Render the full-frame VCPn visual contract without changing inference data."""
    annotated = frame.copy()
    if mask is not None:
        binary = np.asarray(mask, dtype=bool)
        overlay = annotated.copy()
        overlay[binary] = (55, 190, 55)
        annotated = cv2.addWeighted(overlay, 0.38, annotated, 0.62, 0)
        contours, _hierarchy = cv2.findContours(
            binary.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(annotated, contours, -1, (0, 255, 0), 2, cv2.LINE_AA)

    if roi_enabled and roi_px is not None:
        try:
            roi_values = tuple(roi_px)
            if len(roi_values) != 4:
                raise ValueError("ROI deve usar xywh")
            roi_x, roi_y, roi_w, roi_h = (int(round(float(value))) for value in roi_values)
        except (TypeError, ValueError):
            roi_x = roi_y = roi_w = roi_h = 0
        if roi_w > 0 and roi_h > 0:
            roi_bottom_right = (roi_x + roi_w, roi_y + roi_h)
            roi_top_midpoint = (int(round(roi_x + roi_w / 2.0)), roi_y)
            cv2.rectangle(annotated, (roi_x, roi_y), roi_bottom_right, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.drawMarker(
                annotated,
                roi_top_midpoint,
                (255, 0, 255),
                markerType=cv2.MARKER_CROSS,
                markerSize=14,
                thickness=2,
                line_type=cv2.LINE_AA,
            )
            cv2.putText(
                annotated,
                "N",
                (roi_top_midpoint[0] - 5, max(14, roi_top_midpoint[1] - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                annotated,
                "L",
                (roi_top_midpoint[0] + 10, roi_top_midpoint[1] + 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 0),
                2,
                cv2.LINE_AA,
            )

    if x is not None and y is not None and angle_deg is not None:
        center = (int(round(x)), int(round(y)))
        if vcpn_x is not None and vcpn_y is not None:
            tip = (int(round(vcpn_x)), int(round(vcpn_y)))
            if tip != center:
                cv2.arrowedLine(annotated, center, tip, (0, 255, 255), 3, cv2.LINE_AA, tipLength=0.18)
        else:
            radians_value = np.radians(angle_deg)
            direction = np.asarray([np.cos(radians_value), -np.sin(radians_value)])
            tip = tuple(np.rint(np.asarray(center) + direction * 125).astype(int))
            cv2.arrowedLine(annotated, center, tip, (0, 255, 255), 3, cv2.LINE_AA, tipLength=0.18)
        cv2.circle(annotated, center, 6, (255, 255, 255), -1, cv2.LINE_AA)
        if vcpn_x is not None and vcpn_y is not None:
            cv2.circle(annotated, tip, 6, (0, 0, 255), -1, cv2.LINE_AA)

        hud_lines = [
            label,
            f"conf:{int(round((confidence or 0.0) * 100.0))}%",
            "C",
            f"CX:{int(round(x))}",
            f"CY:{int(round(y))}",
            "Vetor",
            f"ang:{int(round(angle_deg))}deg",
            "VCPn",
            f"X:{int(round(vcpn_x if vcpn_x is not None else x))}",
            f"Y:{int(round(vcpn_y if vcpn_y is not None else y))}",
        ]
        if roi_quadrant:
            hud_lines.append(f"Q:{roi_quadrant}")
        if mask_area_cm2 is not None:
            hud_lines.append(f"A:{float(mask_area_cm2):.1f}cm2")

        hud_x, hud_y, line_height = 10, 22, 16
        font_face, font_scale, thickness = cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1
        max_width = max(
            cv2.getTextSize(line, font_face, font_scale, thickness)[0][0]
            for line in hud_lines
        )
        hud = annotated.copy()
        cv2.rectangle(
            hud,
            (hud_x - 6, hud_y - 15),
            (hud_x + max_width + 8, hud_y + line_height * (len(hud_lines) - 1) + 7),
            (10, 18, 10),
            -1,
        )
        annotated = cv2.addWeighted(hud, 0.78, annotated, 0.22, 0)
        for index, line in enumerate(hud_lines):
            color = (255, 255, 255) if line in {"C", "Vetor", "VCPn", label} else (200, 255, 255)
            cv2.putText(
                annotated,
                line,
                (hud_x, hud_y + index * line_height),
                font_face,
                font_scale,
                color,
                thickness,
                cv2.LINE_AA,
            )
    return annotated
