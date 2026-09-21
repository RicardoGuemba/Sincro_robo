from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

# BGR — same pick overlay contract as Realtec Vision Buddmeyer v2108.
PICK_MASK_COLOR_BGR = (0, 255, 128)
PICK_MASK_FILL_ALPHA = 0.40
PICK_MASK_CONTOUR_THICKNESS = 3
PICK_BBOX_THICKNESS = 3
PICK_MAJOR_AXIS_COLOR_BGR = (255, 0, 255)
PICK_ANGLE_REF_COLOR_BGR = (220, 220, 220)
PICK_ANGLE_ARC_COLOR_BGR = (255, 0, 255)
PICK_ARROW_COLOR_BGR = (0, 215, 255)
PICK_CENTROID_COLOR_BGR = (255, 80, 0)
PICK_CENTROID_OUTLINE_BGR = (255, 255, 255)
PICK_CENTROID_RADIUS_PX = 8
ROI_RECT_COLOR_BGR = (0, 255, 0)
ROI_COMPASS_COLOR_BGR = (255, 255, 0)
VCPN_CENTROID_COLOR_BGR = (255, 255, 255)
VCPN_ARROW_COLOR_BGR = (0, 255, 255)
VCPN_POINT_COLOR_BGR = (0, 0, 255)
VCPN_HUD_COLOR_BGR = (255, 255, 255)
VCPN_HUD_OUTLINE_BGR = (0, 0, 0)
VCPN_CENTROID_RADIUS_PX = 6
VCPN_POINT_RADIUS_PX = 6

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
ARROW_LENGTH_PX = 14
ARROW_WIDTH_PX = 10

Point = tuple[int, int]
Segment = tuple[Point, Point]


@dataclass(frozen=True)
class AngleOverlayGeometry:
    magenta_start: Point
    magenta_end: Point
    ref_segments: tuple[Segment, ...]
    arc_points: tuple[Point, ...]
    base_start: Point
    base_end: Point
    arrow_points: tuple[Point, ...]


def aabb_from_mask(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    """Axis-aligned bbox from mask pixels, exclusive x2/y2, same as v2108."""
    binary = np.asarray(mask)
    if binary.dtype == bool:
        active = binary
    elif np.issubdtype(binary.dtype, np.floating):
        active = binary > 0.5
    else:
        active = binary > 0
    ys, xs = np.nonzero(active)
    if xs.size == 0 or ys.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


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
    minor_half_len: float = 0.0,
    scale_x: float = 1.0,
    scale_y: float = 1.0,
    offset_x: float = 0.0,
    offset_y: float = 0.0,
    stroke_scale: float = 1.0,
) -> AngleOverlayGeometry:
    """Magenta major axis, dashed +X reference, mute arc, arrow on the shorter edge."""
    half_major = float(half_len)
    half_minor = float(minor_half_len)
    angle = float(angle_deg or 0.0)
    if half_minor > half_major > 0.0:
        half_major, half_minor = half_minor, half_major
        angle = (angle + 90.0) % 180.0
    rad = math.radians(angle)
    dx = math.cos(rad) * half_major
    dy = math.sin(rad) * half_major
    nx = -math.sin(rad)
    ny = math.cos(rad)
    if half_minor <= 0.0:
        half_minor = ARROW_WIDTH_PX * float(stroke_scale)

    mag_start = _map_point(cx - dx, cy - dy, scale_x, scale_y, offset_x, offset_y)
    mag_end = _map_point(cx + dx, cy + dy, scale_x, scale_y, offset_x, offset_y)
    base_start = _map_point(
        cx + dx - nx * half_minor,
        cy + dy - ny * half_minor,
        scale_x,
        scale_y,
        offset_x,
        offset_y,
    )
    base_end = _map_point(
        cx + dx + nx * half_minor,
        cy + dy + ny * half_minor,
        scale_x,
        scale_y,
        offset_x,
        offset_y,
    )
    base_len = math.hypot(
        float(base_end[0] - base_start[0]),
        float(base_end[1] - base_start[1]),
    )
    arrow_len = min(ARROW_LENGTH_PX * float(stroke_scale), max(3.0, 0.45 * base_len))
    arrow_wid = min(ARROW_WIDTH_PX * float(stroke_scale), max(3.0, 0.6 * base_len))
    arrow_points = _arrowhead_points(base_start, base_end, arrow_len, arrow_wid)

    ref_x1 = min(cx + max(half_major, 0.0), float(frame_w) - 1.0)
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
        half_major,
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
        base_start=base_start,
        base_end=base_end,
        arrow_points=arrow_points,
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


def _arrowhead_points(
    tail: Point,
    tip: Point,
    length_px: float,
    width_px: float,
) -> tuple[Point, ...]:
    dx = float(tip[0] - tail[0])
    dy = float(tip[1] - tail[1])
    hyp = math.hypot(dx, dy)
    if hyp < 1.0 or length_px <= 0.0 or width_px <= 0.0:
        return ()
    ux, uy = dx / hyp, dy / hyp
    px, py = -uy, ux
    base_x = float(tip[0]) - ux * float(length_px)
    base_y = float(tip[1]) - uy * float(length_px)
    half_w = 0.5 * float(width_px)
    left = (int(round(base_x + px * half_w)), int(round(base_y + py * half_w)))
    right = (int(round(base_x - px * half_w)), int(round(base_y - py * half_w)))
    return (tip, left, right)


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
    cv2.line(
        frame,
        geom.base_start,
        geom.base_end,
        PICK_MAJOR_AXIS_COLOR_BGR,
        major_thickness,
        cv2.LINE_AA,
    )
    if len(geom.arrow_points) >= 3:
        head = np.array(geom.arrow_points[:3], dtype=np.int32)
        cv2.fillConvexPoly(frame, head, PICK_ARROW_COLOR_BGR, lineType=cv2.LINE_AA)
        cv2.polylines(
            frame,
            [head.reshape(-1, 1, 2)],
            True,
            PICK_CENTROID_OUTLINE_BGR,
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
    minor_axis_length: float | None = None,
) -> np.ndarray:
    """Pick mask + v2108 AABB + angle overlay. No metric labels."""
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
        box = aabb_from_mask(binary)
        if box is not None:
            x1, y1, x2, y2 = box
            cv2.rectangle(
                annotated,
                (x1, y1),
                (x2, y2),
                PICK_MASK_COLOR_BGR,
                _stroke_px(PICK_BBOX_THICKNESS, scale),
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
    half_major = 0.5 * float(major_axis_length) if major_axis_length and major_axis_length > 0 else 0.0
    half_minor = 0.5 * float(minor_axis_length) if minor_axis_length and minor_axis_length > 0 else 0.0
    if half_major <= 0:
        return annotated
    frame_h, frame_w = annotated.shape[:2]
    geom = compute_angle_overlay_geometry(
        float(x),
        float(y),
        float(angle_deg),
        half_major,
        float(frame_w),
        float(frame_h),
        minor_half_len=half_minor,
        stroke_scale=scale,
    )
    draw_pick_angle_overlay(annotated, geom, scale)
    return annotated


def format_vcpn_hud(
    cx: float,
    cy: float,
    angle_deg: float,
    vcpn_x: float,
    vcpn_y: float,
    *,
    confidence: float | None = None,
    roi_quadrant: str | None = None,
    mask_area_cm2: float | None = None,
    label: str = "Embalagem",
) -> list[str]:
    lines = [label]
    if confidence is not None:
        lines.append(f"conf:{int(round(float(confidence) * 100.0))}%")
    lines.extend(
        [
            "C",
            f"CX:{int(round(float(cx)))}",
            f"CY:{int(round(float(cy)))}",
            "Vetor",
            f"ang:{int(round(float(angle_deg)))}deg",
            "VCPn",
            f"X:{int(round(float(vcpn_x)))}",
            f"Y:{int(round(float(vcpn_y)))}",
        ]
    )
    if roi_quadrant:
        lines.append(f"Q:{roi_quadrant}")
    if mask_area_cm2 is not None:
        lines.append(f"A:{float(mask_area_cm2):.1f}cm2")
    return lines


def _draw_mask_fill(frame: np.ndarray, mask: np.ndarray, stroke_scale: float) -> None:
    import cv2

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
    if not contours:
        return
    overlay = frame.copy()
    cv2.drawContours(overlay, contours, -1, PICK_MASK_COLOR_BGR, thickness=cv2.FILLED)
    cv2.addWeighted(
        overlay,
        PICK_MASK_FILL_ALPHA,
        frame,
        1.0 - PICK_MASK_FILL_ALPHA,
        0,
        frame,
    )
    cv2.drawContours(
        frame,
        contours,
        -1,
        PICK_MASK_COLOR_BGR,
        thickness=_stroke_px(PICK_MASK_CONTOUR_THICKNESS, stroke_scale),
        lineType=cv2.LINE_AA,
    )


def _draw_roi_compass(frame: np.ndarray, roi_px: tuple[float, float, float, float], stroke_scale: float) -> None:
    import cv2
    from .geometry import roi_compass_anchor

    x, y, width, height = roi_px
    thickness = _stroke_px(2, stroke_scale)
    cv2.rectangle(
        frame,
        (int(round(x)), int(round(y))),
        (int(round(x + width)), int(round(y + height))),
        ROI_RECT_COLOR_BGR,
        thickness,
        cv2.LINE_AA,
    )
    anchor_x, anchor_y = roi_compass_anchor(roi_px)
    ax, ay = int(round(anchor_x)), int(round(anchor_y))
    arm = _stroke_px(18, stroke_scale)
    cv2.line(frame, (ax, ay), (ax, ay - arm), ROI_COMPASS_COLOR_BGR, thickness, cv2.LINE_AA)
    cv2.line(frame, (ax, ay), (ax + arm, ay), ROI_COMPASS_COLOR_BGR, thickness, cv2.LINE_AA)
    cv2.drawMarker(
        frame,
        (ax, ay),
        ROI_COMPASS_COLOR_BGR,
        markerType=cv2.MARKER_CROSS,
        markerSize=_stroke_px(10, stroke_scale),
        thickness=thickness,
        line_type=cv2.LINE_AA,
    )
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.45 * float(stroke_scale)
    cv2.putText(
        frame, "N", (ax - 6, ay - arm - 4), font, font_scale, ROI_COMPASS_COLOR_BGR, 1, cv2.LINE_AA
    )
    cv2.putText(
        frame, "L", (ax + arm + 4, ay + 4), font, font_scale, ROI_COMPASS_COLOR_BGR, 1, cv2.LINE_AA
    )


def _draw_vcpn_arrow(frame: np.ndarray, cx: float, cy: float, vx: float, vy: float, stroke_scale: float) -> None:
    import cv2

    start = (int(round(cx)), int(round(cy)))
    end = (int(round(vx)), int(round(vy)))
    if start == end:
        return
    thickness = _stroke_px(2, stroke_scale)
    cv2.line(frame, start, end, VCPN_ARROW_COLOR_BGR, thickness, cv2.LINE_AA)
    head = _arrowhead_points(
        start,
        end,
        ARROW_LENGTH_PX * float(stroke_scale),
        ARROW_WIDTH_PX * float(stroke_scale),
    )
    if len(head) >= 3:
        cv2.fillConvexPoly(frame, np.array(head[:3], dtype=np.int32), VCPN_ARROW_COLOR_BGR)


def _draw_hud_lines(frame: np.ndarray, lines: list[str]) -> None:
    import cv2

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.45
    x, y = 8, 18
    for line in lines:
        cv2.putText(frame, line, (x, y), font, scale, VCPN_HUD_OUTLINE_BGR, 3, cv2.LINE_AA)
        cv2.putText(frame, line, (x, y), font, scale, VCPN_HUD_COLOR_BGR, 1, cv2.LINE_AA)
        y += 16


def annotate_vcpn_overlay(
    frame: np.ndarray,
    mask: np.ndarray | None,
    x: float | None = None,
    y: float | None = None,
    angle_deg: float | None = None,
    *,
    vcpn_x: float | None = None,
    vcpn_y: float | None = None,
    roi_px: tuple[float, float, float, float] | None = None,
    roi_enabled: bool = True,
    roi_quadrant: str | None = None,
    confidence: float | None = None,
    mask_area_cm2: float | None = None,
    label: str = "Embalagem",
    stroke_scale: float = 1.0,
) -> np.ndarray:
    """Mask, ROI, north compass, C, C→VCPn arrow, VCPn, and ASCII HUD."""
    import cv2

    annotated = frame.copy()
    scale = float(stroke_scale) if stroke_scale > 0 else 1.0
    if mask is not None:
        _draw_mask_fill(annotated, mask, scale)
    if roi_enabled and roi_px is not None:
        _draw_roi_compass(annotated, roi_px, scale)
    if x is None or y is None:
        return annotated
    cx, cy = float(x), float(y)
    vx = float(vcpn_x) if vcpn_x is not None else cx
    vy = float(vcpn_y) if vcpn_y is not None else cy
    centroid_r = _stroke_px(VCPN_CENTROID_RADIUS_PX, scale)
    cv2.circle(
        annotated,
        (int(round(cx)), int(round(cy))),
        centroid_r,
        VCPN_CENTROID_COLOR_BGR,
        thickness=-1,
        lineType=cv2.LINE_AA,
    )
    _draw_vcpn_arrow(annotated, cx, cy, vx, vy, scale)
    point_r = _stroke_px(VCPN_POINT_RADIUS_PX, scale)
    cv2.circle(
        annotated,
        (int(round(vx)), int(round(vy))),
        point_r,
        VCPN_POINT_COLOR_BGR,
        thickness=-1,
        lineType=cv2.LINE_AA,
    )
    if angle_deg is not None:
        _draw_hud_lines(
            annotated,
            format_vcpn_hud(
                cx,
                cy,
                angle_deg,
                vx,
                vy,
                confidence=confidence,
                roi_quadrant=roi_quadrant,
                mask_area_cm2=mask_area_cm2,
                label=label,
            ),
        )
    return annotated

