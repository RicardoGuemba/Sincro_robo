from __future__ import annotations

import math

import numpy as np
import pytest

from sincro_robo.adapters.segmenter import annotate_frame
from sincro_robo.geometry import MoldPoseEstimator
from sincro_robo.overlay import (
    MIN_ARC_RADIUS_PX,
    PICK_ANGLE_REF_COLOR_BGR,
    PICK_CENTROID_COLOR_BGR,
    PICK_MAJOR_AXIS_COLOR_BGR,
    PICK_MASK_COLOR_BGR,
    compute_angle_overlay_geometry,
    overlay_stroke_scale,
)


def _color_hits(patch: np.ndarray, color) -> bool:
    target = np.asarray(color, dtype=np.uint8)
    if patch.ndim == 1:
        return bool(np.array_equal(patch, target))
    return bool(np.any(np.all(patch == target, axis=-1)))


def _row_has_color(img: np.ndarray, y: int, x0: int, x1: int, color) -> bool:
    lo, hi = sorted((int(x0), int(x1)))
    return _color_hits(img[int(y), lo : hi + 1], color)


def _col_has_color(img: np.ndarray, x: int, y0: int, y1: int, color) -> bool:
    lo, hi = sorted((int(y0), int(y1)))
    return _color_hits(img[lo : hi + 1, int(x)], color)


def _neighborhood_has_color(img: np.ndarray, y: int, x: int, color, r: int = 2) -> bool:
    height, width = img.shape[:2]
    y0, y1 = max(0, y - r), min(height, y + r + 1)
    x0, x1 = max(0, x - r), min(width, x + r + 1)
    return _color_hits(img[y0:y1, x0:x1], color)


def test_native_frame_stroke_scale_is_inverse_of_pick_map() -> None:
    assert overlay_stroke_scale(10 / 27, 10 / 27) == pytest.approx(2.7)
    assert overlay_stroke_scale(1.0, 1.0) == pytest.approx(1.0)


def test_geometry_horizontal_ref_aligned_with_magenta_arc_omitted() -> None:
    geom = compute_angle_overlay_geometry(100.0, 100.0, 0.0, 40.0, 200.0, 200.0)
    assert geom.magenta_start[1] == geom.magenta_end[1] == 100
    assert geom.magenta_end[0] > 100
    assert geom.magenta_start[0] < 100
    assert geom.arc_points == ()
    assert geom.ref_segments
    for (x0, y0), (x1, y1) in geom.ref_segments:
        assert y0 == 100 and y1 == 100
        assert x0 >= 100 and x1 >= 100


def test_geometry_90deg_arc_is_downward_quadrant() -> None:
    geom = compute_angle_overlay_geometry(100.0, 100.0, 90.0, 40.0, 200.0, 200.0)
    assert geom.magenta_end[0] == 100
    assert geom.magenta_end[1] > 100
    assert len(geom.arc_points) >= 2
    first, last = geom.arc_points[0], geom.arc_points[-1]
    assert first[0] > 100
    assert abs(first[1] - 100) <= 2
    assert last[1] > 100
    assert abs(last[0] - 100) <= 2
    mid = geom.arc_points[len(geom.arc_points) // 2]
    assert mid[0] > 100
    assert mid[1] > 100
    ys = [y for _, y in geom.arc_points]
    assert min(ys) >= 100 - 2


def test_geometry_135deg_tip_lower_left_arc_passes_down() -> None:
    geom = compute_angle_overlay_geometry(100.0, 100.0, 135.0, 40.0, 200.0, 200.0)
    assert geom.magenta_end[0] < 100
    assert geom.magenta_end[1] > 100
    assert len(geom.arc_points) >= 2
    first, last = geom.arc_points[0], geom.arc_points[-1]
    assert first[0] > 100
    assert abs(first[1] - 100) <= 2
    assert last[0] < 100
    assert last[1] > 100
    down = min(geom.arc_points, key=lambda point: abs(point[0] - 100))
    assert down[1] > 100
    assert abs(down[0] - 100) <= 3


def test_geometry_shrinks_or_omits_arc_near_frame_edge() -> None:
    roomy = compute_angle_overlay_geometry(100.0, 100.0, 90.0, 40.0, 200.0, 200.0)
    tight = compute_angle_overlay_geometry(195.0, 100.0, 90.0, 40.0, 200.0, 200.0)
    assert roomy.arc_points
    if tight.arc_points:
        xs = [point[0] for point in tight.arc_points]
        ys = [point[1] for point in tight.arc_points]
        assert max(xs) <= 198
        assert max(ys) <= 198
        span = max(math.hypot(point[0] - 195, point[1] - 100) for point in tight.arc_points)
        assert span <= MIN_ARC_RADIUS_PX + 4
    assert tight.arc_points == () or len(tight.arc_points) >= 2


def test_overlay_horizontal_magenta_and_no_downward_arc() -> None:
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    mask = np.zeros((200, 200), dtype=bool)
    mask[60:140, 60:140] = True
    out = annotate_frame(
        frame,
        mask,
        100.0,
        100.0,
        0.0,
        major_axis_length=80.0,
    )
    assert _row_has_color(out, 100, 118, 138, PICK_MAJOR_AXIS_COLOR_BGR)
    assert not _neighborhood_has_color(out, 118, 118, PICK_MAJOR_AXIS_COLOR_BGR, r=2)
    assert not _col_has_color(out, 100, 118, 138, PICK_ANGLE_REF_COLOR_BGR)


def test_overlay_90deg_has_ref_magenta_and_downward_arc() -> None:
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    mask = np.zeros((200, 200), dtype=bool)
    mask[60:140, 60:140] = True
    out = annotate_frame(
        frame,
        mask,
        100.0,
        100.0,
        90.0,
        major_axis_length=80.0,
    )
    assert _row_has_color(out, 100, 112, 138, PICK_ANGLE_REF_COLOR_BGR)
    assert _col_has_color(out, 100, 118, 138, PICK_MAJOR_AXIS_COLOR_BGR)
    assert _neighborhood_has_color(out, 113, 113, PICK_MAJOR_AXIS_COLOR_BGR, r=3)


def test_pick_centroid_is_blue_and_mask_is_teal() -> None:
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    mask = np.zeros((200, 200), dtype=bool)
    mask[60:140, 60:140] = True
    out = annotate_frame(frame, mask, 20.0, 20.0)
    assert np.array_equal(out[20, 20], PICK_CENTROID_COLOR_BGR)
    blended = out[100, 100]
    assert blended[1] > blended[0] and blended[1] > blended[2]
    assert not np.array_equal(blended, PICK_MASK_COLOR_BGR)


def test_overlay_has_no_theta_label() -> None:
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    out = annotate_frame(frame, None, 100.0, 100.0, 37.0, major_axis_length=40.0)
    # Old SINCRO label sat near (114, 86) in orange/teal text.
    assert np.array_equal(out[86, 114], (0, 0, 0))
    colors = {tuple(int(channel) for channel in color) for color in np.unique(out.reshape(-1, 3), axis=0)}
    assert (40, 183, 255) not in colors
    assert (71, 231, 186) not in colors
    assert PICK_CENTROID_COLOR_BGR in colors
    assert PICK_MAJOR_AXIS_COLOR_BGR in colors


def test_estimator_length_feeds_overlay_axis() -> None:
    mask = np.zeros((200, 200), dtype=np.uint8)
    mask[90:110, 20:180] = 1
    pose = MoldPoseEstimator().estimate(mask)
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    out = annotate_frame(
        frame,
        mask.astype(bool),
        pose.x,
        pose.y,
        pose.angle_deg,
        major_axis_length=pose.major_axis_length,
    )
    assert pose.major_axis_length == pytest.approx(160.0, abs=2.0)
    assert _row_has_color(out, int(round(pose.y)), 30, 50, PICK_MAJOR_AXIS_COLOR_BGR)
