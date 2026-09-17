from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

# BGR — same pick overlay contract as Realtec Vision Buddmeyer v2108.
PICK_MASK_COLOR_BGR = (0, 255, 128)
PICK_MASK_FILL_ALPHA = 0.40
PICK_MASK_CONTOUR_THICKNESS = 3
PICK_MAJOR_AXIS_COLOR_BGR = (255, 0, 255)
PICK_ANGLE_REF_COLOR_BGR = (220, 220, 220)
PICK_ANGLE_ARC_COLOR_BGR = (255, 0, 255)
PICK_CENTROID_COLOR_BGR = (255, 80, 0)
PICK_CENTROID_OUTLINE_BGR = (255, 255, 255)
PICK_CENTROID_RADIUS_PX = 8

NEAR_ZERO_ANGLE_DEG = 2.0
ARC_RADIUS_FRACTION = 0.45
MIN_ARC_RADIUS_PX = 8.0
ARC_STEP_DEG = 4.0
FRAME_MARGIN_PX = 2.0
DASH_PX = 8.0
GAP_PX = 6.0
MAJOR_AXIS_THICKNESS = 3
REF_AXIS_THICKNESS = 2
ARC_THICKNESS = 2
TIP_CIRCLE_RADIUS_PX = 5

Point = tuple[int, int]
Segment = tuple[Point, Point]


@dataclass(frozen=True)
class AngleOverlayGeometry:
    magenta_start: Point
    magenta_end: Point
    ref_segments: tuple[Segment, ...]
    arc_points: tuple[Point, ...]


def overlay_stroke_scale(scale_x: float, scale_y: float) -> float:
    """Scale v2108 pick-space stroke lengths onto a native camera frame."""
    sx = float(scale_x)
    sy = float(scale_y)
    if sx <= 0 or abs(sx - sy) > 1e-6:
        return 1.0
    return 1.0 / sx


def _stroke_px(value: float, stroke_scale: float) -> int:
    return max(1, int(round(float(value) * float(stroke_scale))))


def compute_angle_overlay_geometry(
    cx: float,
    cy: float,
    angle_deg: float,
    half_len: float,
    frame_w: float,
    frame_h: float,
    *,
    scale_x: float = 1.0,
    scale_y: float = 1.0,
    offset_x: float = 0.0,
    offset_y: float = 0.0,
    stroke_scale: float = 1.0,
) -> AngleOverlayGeometry:
    """Magenta major axis, dashed +X reference, mute arc. Image +X right, +Y down."""
    half = float(half_len)
    angle = float(angle_deg or 0.0)
    rad = math.radians(angle)
    dx = math.cos(rad) * half
    dy = math.sin(rad) * half

    mag_start = _map_point(cx - dx, cy - dy, scale_x, scale_y, offset_x, offset_y)
    mag_end = _map_point(cx + dx, cy + dy, scale_x, scale_y, offset_x, offset_y)

    ref_x1 = min(cx + max(half, 0.0), float(frame_w) - 1.0)
    ref_segments: tuple[Segment, ...] = ()
    if ref_x1 > cx + 0.5:
        ref_segments = _dashed_segments(
            cx,
            cy,
            ref_x1,
            cy,
            scale_x,
            scale_y,
            offset_x,
            offset_y,
            stroke_scale,
        )

    arc_points = _arc_points(
        cx,
        cy,
        angle,
        half,
        frame_w,
        frame_h,
        scale_x,
        scale_y,
        offset_x,
        offset_y,
        stroke_scale,
    )
    return AngleOverlayGeometry(
        magenta_start=mag_start,
        magenta_end=mag_end,
        ref_segments=ref_segments,
        arc_points=arc_points,
    )


def _map_point(
    x: float,
    y: float,
    scale_x: float,
    scale_y: float,
    offset_x: float,
    offset_y: float,
) -> Point:
    return (int(x * scale_x) + int(offset_x), int(y * scale_y) + int(offset_y))


def _dashed_segments(
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    scale_x: float,
    scale_y: float,
    offset_x: float,
    offset_y: float,
    stroke_scale: float,
) -> tuple[Segment, ...]:
    dx = x1 - x0
    dy = y1 - y0
    length = math.hypot(dx, dy)
    if length < 1.0:
        return ()
    ux, uy = dx / length, dy / length
    dash = DASH_PX * float(stroke_scale)
    gap = GAP_PX * float(stroke_scale)
    segs = []
    pos = 0.0
    while pos < length:
        end = min(pos + dash, length)
        p0 = _map_point(
            x0 + ux * pos, y0 + uy * pos, scale_x, scale_y, offset_x, offset_y
        )
        p1 = _map_point(
            x0 + ux * end, y0 + uy * end, scale_x, scale_y, offset_x, offset_y
        )
        if p0 != p1:
            segs.append((p0, p1))
        pos += dash + gap
    return tuple(segs)


def _max_arc_radius(
    cx: float,
    cy: float,
    angle_deg: float,
    frame_w: float,
    frame_h: float,
    margin: float = FRAME_MARGIN_PX,
) -> float:
    if angle_deg <= 0.0:
        return 0.0
    theta = math.radians(angle_deg)
    cos_min = math.cos(theta)
    sin_max = math.sin(min(theta, math.pi / 2.0))

    r_allowed = float(frame_w) - margin - cx
    if cos_min < 0.0:
        left = (cx - margin) / (-cos_min)
        r_allowed = min(r_allowed, left)
    if sin_max > 0.0:
        r_allowed = min(r_allowed, (float(frame_h) - margin - cy) / sin_max)
    if cy < margin:
        return 0.0
    return max(0.0, r_allowed)


def _arc_points(
    cx: float,
    cy: float,
    angle_deg: float,
    half_len: float,
    frame_w: float,
    frame_h: float,
    scale_x: float,
    scale_y: float,
    offset_x: float,
    offset_y: float,
    stroke_scale: float,
) -> tuple[Point, ...]:
    if angle_deg < NEAR_ZERO_ANGLE_DEG:
        return ()
    desired = ARC_RADIUS_FRACTION * max(half_len, 0.0)
    allowed = _max_arc_radius(cx, cy, angle_deg, frame_w, frame_h)
    radius = min(desired, allowed)
    if radius < MIN_ARC_RADIUS_PX * float(stroke_scale):
        return ()

    n = max(2, int(math.ceil(angle_deg / ARC_STEP_DEG)))
    pts = []
    last: Optional[Point] = None
    for i in range(n + 1):
        t = angle_deg * (i / n)
        rad = math.radians(t)
        pt = _map_point(
            cx + radius * math.cos(rad),
            cy + radius * math.sin(rad),
            scale_x,
            scale_y,
            offset_x,
            offset_y,
        )
        if pt != last:
            pts.append(pt)
            last = pt
    if len(pts) < 2:
        return ()
    return tuple(pts)


def draw_pick_angle_overlay(
    frame: np.ndarray,
    geom: AngleOverlayGeometry,
    stroke_scale: float = 1.0,
) -> None:
    import cv2

    ref_thickness = _stroke_px(REF_AXIS_THICKNESS, stroke_scale)
    arc_thickness = _stroke_px(ARC_THICKNESS, stroke_scale)
    major_thickness = _stroke_px(MAJOR_AXIS_THICKNESS, stroke_scale)
    tip_radius = _stroke_px(TIP_CIRCLE_RADIUS_PX, stroke_scale)
    for (x0, y0), (x1, y1) in geom.ref_segments:
        cv2.line(frame, (x0, y0), (x1, y1), PICK_ANGLE_REF_COLOR_BGR, ref_thickness)
    if len(geom.arc_points) >= 2:
        pts = np.array(geom.arc_points, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(frame, [pts], False, PICK_ANGLE_ARC_COLOR_BGR, arc_thickness)
    cv2.line(
        frame,
        geom.magenta_start,
        geom.magenta_end,
        PICK_MAJOR_AXIS_COLOR_BGR,
        major_thickness,
        cv2.LINE_AA,
    )
    cv2.circle(
        frame,
        geom.magenta_end,
        tip_radius,
        PICK_MAJOR_AXIS_COLOR_BGR,
        1,
        cv2.LINE_AA,
    )


def annotate_pick_overlay(
    frame: np.ndarray,
    mask: np.ndarray | None,
    x: float | None = None,
    y: float | None = None,
    angle_deg: float | None = None,
    major_axis_length: float | None = None,
    stroke_scale: float = 1.0,
) -> np.ndarray:
    """Pick mask + v2108 angle overlay. No metric labels."""
    import cv2

    annotated = frame.copy()
    scale = float(stroke_scale) if stroke_scale > 0 else 1.0
    if mask is not None:
        binary = np.asarray(mask)
        if binary.dtype == bool:
            bin_mask = binary.astype(np.uint8) * 255
        elif np.issubdtype(binary.dtype, np.floating):
            bin_mask = (binary > 0.5).astype(np.uint8) * 255
        else:
            bin_mask = (binary > 0).astype(np.uint8) * 255
        contours, _hierarchy = cv2.findContours(
            bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if contours:
            overlay = annotated.copy()
            cv2.drawContours(overlay, contours, -1, PICK_MASK_COLOR_BGR, thickness=cv2.FILLED)
            cv2.addWeighted(
                overlay,
                PICK_MASK_FILL_ALPHA,
                annotated,
                1.0 - PICK_MASK_FILL_ALPHA,
                0,
                annotated,
            )
            cv2.drawContours(
                annotated,
                contours,
                -1,
                PICK_MASK_COLOR_BGR,
                thickness=_stroke_px(PICK_MASK_CONTOUR_THICKNESS, scale),
                lineType=cv2.LINE_AA,
            )
    if x is None or y is None:
        return annotated
    cx, cy = int(round(float(x))), int(round(float(y)))
    centroid_r = _stroke_px(PICK_CENTROID_RADIUS_PX, scale)
    cv2.circle(annotated, (cx, cy), centroid_r, PICK_CENTROID_COLOR_BGR, thickness=-1, lineType=cv2.LINE_AA)
    cv2.circle(
        annotated,
        (cx, cy),
        centroid_r,
        PICK_CENTROID_OUTLINE_BGR,
        1,
        cv2.LINE_AA,
    )
    if angle_deg is None:
        return annotated
    half = 0.5 * float(major_axis_length) if major_axis_length and major_axis_length > 0 else 0.0
    if half <= 0:
        return annotated
    frame_h, frame_w = annotated.shape[:2]
    geom = compute_angle_overlay_geometry(
        float(x),
        float(y),
        float(angle_deg),
        half,
        float(frame_w),
        float(frame_h),
        stroke_scale=scale,
    )
    draw_pick_angle_overlay(annotated, geom, scale)
    return annotated
