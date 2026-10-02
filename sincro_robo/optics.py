"""Geometria de câmera do guia OpenCV: intrínseca, extrínseca e validação.

As funções são puras. A thread da câmera e o assistente chamam daqui; nenhuma
delas abre dispositivo nem escreve no CLP.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

import cv2
import numpy as np

PATTERN = (9, 6)
MIN_ACCEPTED = 20
REPROJECTION_LIMIT_PX = 1.0
MIN_FRAME_AREA_MM2 = 500.0
VALIDATION_PLANES_MM = (0.0, 200.0, 400.0)

CELL_NAMES = {
    (0, 0): "canto superior esquerdo",
    (1, 0): "borda superior",
    (2, 0): "canto superior direito",
    (0, 1): "borda esquerda",
    (1, 1): "centro",
    (2, 1): "borda direita",
    (0, 2): "canto inferior esquerdo",
    (1, 2): "borda inferior",
    (2, 2): "canto inferior direito",
}


def object_points(pattern: tuple[int, int] = PATTERN, square_size_mm: float = 30.0) -> np.ndarray:
    grid = np.mgrid[0 : pattern[0], 0 : pattern[1]].T.reshape(-1, 2)
    points = np.zeros((pattern[0] * pattern[1], 3), np.float32)
    points[:, :2] = grid
    points *= float(square_size_mm)
    return points


def _as_matrix(values: Any, shape: tuple[int, int]) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.shape != shape:
        raise ValueError(f"Matriz deveria ter forma {shape}")
    if not np.isfinite(matrix).all():
        raise ValueError("Matriz contém valor não finito")
    return matrix


def detect_chessboard(frame_bgr: np.ndarray, pattern: tuple[int, int] = PATTERN) -> dict[str, Any]:
    """Detecta cantos internos no frame BGR de produção, com refinamento subpixel."""
    if frame_bgr.ndim != 3:
        raise ValueError("O frame do tabuleiro precisa ser BGR")
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape[:2]
    found = False
    corners: np.ndarray | None = None
    if hasattr(cv2, "findChessboardCornersSB"):
        flags = cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
        found, corners = cv2.findChessboardCornersSB(gray, pattern, flags)
    if not found or corners is None:
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        found, corners = cv2.findChessboardCorners(gray, pattern, flags)
        if found and corners is not None:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001)
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    if not found or corners is None:
        return {"found": False, "count": 0, "corners": [], "cell": None, "width": width, "height": height}
    points = np.asarray(corners, dtype=np.float64).reshape(-1, 2)
    cell = coverage_cell(points, width, height)
    return {
        "found": points.shape[0] == pattern[0] * pattern[1],
        "count": int(points.shape[0]),
        "corners": points.tolist(),
        "cell": cell,
        "width": int(width),
        "height": int(height),
    }


def coverage_cell(corners: np.ndarray, width: int, height: int) -> list[int]:
    center = np.asarray(corners, dtype=np.float64).reshape(-1, 2).mean(axis=0)
    col = int(np.clip(center[0] / max(width, 1) * 3.0, 0, 2.999))
    row = int(np.clip(center[1] / max(height, 1) * 3.0, 0, 2.999))
    return [col, row]


def missing_cells(filled: set[tuple[int, int]]) -> list[str]:
    ordered = [(1, 1), (0, 0), (2, 0), (0, 2), (2, 2), (1, 0), (0, 1), (2, 1), (1, 2)]
    return [CELL_NAMES[cell] for cell in ordered if cell not in filled]


def suggestion_for(filled: set[tuple[int, int]]) -> str:
    missing = missing_cells(filled)
    if not missing:
        return "Cobertura completa. Pode avançar quando houver imagens aceitas suficientes."
    return f"Mover o tabuleiro para: {missing[0]}."


def draw_detection(frame_bgr: np.ndarray, detection: dict[str, Any], pattern: tuple[int, int] = PATTERN) -> np.ndarray:
    annotated = frame_bgr.copy()
    corners = detection.get("corners") or []
    if detection.get("found") and corners:
        array = np.asarray(corners, dtype=np.float32).reshape(-1, 1, 2)
        cv2.drawChessboardCorners(annotated, pattern, array, True)
        origin = tuple(int(round(value)) for value in corners[0])
        cv2.circle(annotated, origin, 8, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.putText(
            annotated, "origem", (origin[0] + 10, origin[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA,
        )
    else:
        cv2.putText(
            annotated, "tabuleiro nao detectado", (24, 36),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 80, 255), 2, cv2.LINE_AA,
        )
    return annotated


def calibrate_intrinsic(
    image_points: list[list[list[float]]],
    image_size: tuple[int, int],
    square_size_mm: float,
    pattern: tuple[int, int] = PATTERN,
) -> dict[str, Any]:
    if len(image_points) < 3:
        raise ValueError("São necessárias pelo menos 3 imagens para estimar K e a distorção")
    expected = pattern[0] * pattern[1]
    obj = object_points(pattern, square_size_mm)
    objects: list[np.ndarray] = []
    images: list[np.ndarray] = []
    for corners in image_points:
        array = np.asarray(corners, dtype=np.float32).reshape(-1, 2)
        if array.shape[0] != expected:
            raise ValueError(f"Cada imagem aceita precisa ter {expected} cantos")
        objects.append(obj.copy())
        images.append(array.reshape(-1, 1, 2))
    width, height = int(image_size[0]), int(image_size[1])
    rms, matrix, dist, rvecs, tvecs = cv2.calibrateCamera(objects, images, (width, height), None, None)
    errors = [
        reprojection_error(obj, images[index], rvecs[index], tvecs[index], matrix, dist)
        for index in range(len(images))
    ]
    dist_list = np.asarray(dist, dtype=np.float64).reshape(-1).tolist()
    new_matrix, roi = cv2.getOptimalNewCameraMatrix(matrix, dist, (width, height), 0, (width, height))
    return {
        "rms_px": float(rms),
        "camera_matrix": np.asarray(matrix, dtype=np.float64).tolist(),
        "dist_coeffs": [float(value) for value in dist_list],
        "rectified_camera_matrix": np.asarray(new_matrix, dtype=np.float64).tolist(),
        "valid_roi_px": [int(value) for value in roi],
        "per_image_px": [float(value) for value in errors],
        "image_size_px": [width, height],
        "model": "opencv_standard_5",
    }


def reprojection_error(
    obj_points: np.ndarray,
    image_points: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> float:
    projected, _ = cv2.projectPoints(obj_points, rvec, tvec, camera_matrix, dist_coeffs)
    observed = np.asarray(image_points, dtype=np.float64).reshape(-1, 2)
    predicted = np.asarray(projected, dtype=np.float64).reshape(-1, 2)
    return float(cv2.norm(observed, predicted, cv2.NORM_L2) / max(len(predicted), 1))


def undistort_points(
    points: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    rectified_matrix: np.ndarray,
) -> np.ndarray:
    source = np.asarray(points, dtype=np.float64).reshape(-1, 1, 2)
    corrected = cv2.undistortPoints(
        source,
        _as_matrix(camera_matrix, (3, 3)),
        np.asarray(dist_coeffs, dtype=np.float64).reshape(-1, 1),
        R=np.eye(3),
        P=_as_matrix(rectified_matrix, (3, 3)),
    )
    return np.asarray(corrected, dtype=np.float64).reshape(-1, 2)


def undistort_pairs(pairs: list[dict[str, Any]], profile: dict[str, Any]) -> list[dict[str, Any]]:
    """Devolve cópias dos pares com o centroide no espaço retificado. O par gravado não muda."""
    intrinsic = profile.get("intrinsic") or profile
    matrix = intrinsic.get("camera_matrix")
    dist = intrinsic.get("dist_coeffs")
    rectified = intrinsic.get("rectified_camera_matrix")
    if matrix is None or dist is None or rectified is None:
        raise ValueError("Perfil óptico sem K, distorção ou K retificada")
    prepared: list[dict[str, Any]] = []
    for pair in pairs:
        vision = dict(pair["vision"])
        corrected = undistort_points(
            np.asarray([[vision["x"], vision["y"]]], dtype=np.float64),
            matrix,
            dist,
            rectified,
        )[0]
        vision["x"] = float(corrected[0])
        vision["y"] = float(corrected[1])
        prepared.append({**pair, "vision": vision})
    return prepared


def preview_before_after(frame_bgr: np.ndarray, camera_matrix: np.ndarray, dist_coeffs: np.ndarray) -> np.ndarray:
    undistorted = cv2.undistort(frame_bgr, _as_matrix(camera_matrix, (3, 3)), np.asarray(dist_coeffs, dtype=np.float64))
    height, width = frame_bgr.shape[:2]
    left = cv2.resize(frame_bgr, (width // 2, height // 2), interpolation=cv2.INTER_AREA)
    right = cv2.resize(undistorted, (width // 2, height // 2), interpolation=cv2.INTER_AREA)
    canvas = np.hstack([left, right])
    cv2.putText(canvas, "ANTES", (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, "DEPOIS", (width // 2 + 16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return canvas


def board_frame(origin_mm: list[float], plus_x_mm: list[float], plus_y_mm: list[float]) -> dict[str, Any]:
    """Base do tabuleiro no referencial do robô, a partir de três toques do TCP."""
    origin = np.asarray(origin_mm, dtype=np.float64).reshape(3)
    axis_x = np.asarray(plus_x_mm, dtype=np.float64).reshape(3) - origin
    axis_y = np.asarray(plus_y_mm, dtype=np.float64).reshape(3) - origin
    if float(np.linalg.norm(axis_x)) < 10.0 or float(np.linalg.norm(axis_y)) < 10.0:
        raise ValueError("Os pontos de eixo precisam ficar a mais de 10 mm da origem")
    normal = np.cross(axis_x, axis_y)
    if float(np.linalg.norm(normal)) < MIN_FRAME_AREA_MM2:
        raise ValueError("Os eixos estão quase alinhados. Afaste o ponto Y da reta X.")
    x_unit = axis_x / np.linalg.norm(axis_x)
    y_orth = axis_y - x_unit * float(np.dot(axis_y, x_unit))
    y_unit = y_orth / np.linalg.norm(y_orth)
    z_unit = np.cross(x_unit, y_unit)
    rotation = np.column_stack([x_unit, y_unit, z_unit])
    return {"R": rotation.tolist(), "t": origin.tolist()}


def solve_extrinsic(
    corners: list[list[float]],
    camera_matrix: list[list[float]],
    dist_coeffs: list[float],
    square_size_mm: float,
    pattern: tuple[int, int] = PATTERN,
) -> dict[str, Any]:
    obj = object_points(pattern, square_size_mm)
    image = np.asarray(corners, dtype=np.float64).reshape(-1, 1, 2)
    if image.shape[0] != obj.shape[0]:
        raise ValueError("A imagem extrínseca não tem os 54 cantos")
    ok, rvec, tvec = cv2.solvePnP(
        obj,
        image,
        _as_matrix(camera_matrix, (3, 3)),
        np.asarray(dist_coeffs, dtype=np.float64).reshape(-1, 1),
    )
    if not ok:
        raise ValueError("solvePnP não convergiu nesta imagem")
    rotation, _ = cv2.Rodrigues(rvec)
    translation = np.asarray(tvec, dtype=np.float64).reshape(3)
    inverse_r = rotation.T
    inverse_t = -inverse_r @ translation
    return {
        "object_to_camera": {"R": rotation.tolist(), "t": translation.tolist()},
        "camera_to_object": {"R": inverse_r.tolist(), "t": inverse_t.tolist()},
    }


def robot_from_camera(object_to_camera: dict[str, Any], board_to_robot: dict[str, Any]) -> dict[str, Any]:
    """P_camera = R_rc · P_robot + t_rc, com a câmera fixa e o tabuleiro conhecido no robô."""
    rotation_bc = _as_matrix(object_to_camera["R"], (3, 3))
    translation_bc = np.asarray(object_to_camera["t"], dtype=np.float64).reshape(3)
    rotation_br = _as_matrix(board_to_robot["R"], (3, 3))
    translation_br = np.asarray(board_to_robot["t"], dtype=np.float64).reshape(3)
    rotation_rc = rotation_bc @ rotation_br.T
    translation_rc = translation_bc - rotation_rc @ translation_br
    return {"R": rotation_rc.tolist(), "t": translation_rc.tolist()}


def corner_in_robot(board_to_robot: dict[str, Any], board_point_mm: list[float]) -> list[float]:
    rotation = _as_matrix(board_to_robot["R"], (3, 3))
    origin = np.asarray(board_to_robot["t"], dtype=np.float64).reshape(3)
    point = origin + rotation @ np.asarray(board_point_mm, dtype=np.float64).reshape(3)
    return [float(value) for value in point]


def direction_errors(
    tcp_mm: list[float],
    board_to_robot: dict[str, Any],
    square_size_mm: float,
) -> dict[str, Any]:
    """Compara o TCP num canto com a previsão do eixo ensinado e com o eixo invertido."""
    correct = np.asarray(corner_in_robot(board_to_robot, [square_size_mm, 0.0, 0.0]), dtype=np.float64)
    inverted_frame = {
        "R": (-_as_matrix(board_to_robot["R"], (3, 3))).tolist(),
        "t": board_to_robot["t"],
    }
    inverted = np.asarray(corner_in_robot(inverted_frame, [square_size_mm, 0.0, 0.0]), dtype=np.float64)
    tcp = np.asarray(tcp_mm, dtype=np.float64).reshape(3)
    return {
        "predicted_mm": correct.tolist(),
        "inverted_mm": inverted.tolist(),
        "error_correct_mm": float(np.linalg.norm(tcp - correct)),
        "error_inverted_mm": float(np.linalg.norm(tcp - inverted)),
    }


def _stats(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {"mae": 0.0, "rmse": 0.0, "max": 0.0}
    return {
        "mae": float(np.mean(np.abs(array))),
        "rmse": float(np.sqrt(np.mean(np.square(array)))),
        "max": float(np.max(np.abs(array))),
    }


def validation_metrics(
    detected_corners: list[list[float]],
    board_to_robot: dict[str, Any],
    robot_from_camera_pose: dict[str, Any],
    camera_matrix: list[list[float]],
    dist_coeffs: list[float],
    square_size_mm: float,
    plane_z_mm: float,
    pattern: tuple[int, int] = PATTERN,
) -> dict[str, Any]:
    """Projeta o tabuleiro no plano Z com a extrínseca já aceita. Não recalcula K nem R,t."""
    obj = object_points(pattern, square_size_mm)
    rotation = _as_matrix(board_to_robot["R"], (3, 3))
    origin = np.asarray(board_to_robot["t"], dtype=np.float64).reshape(3)
    lift = np.array([0.0, 0.0, float(plane_z_mm) - float(origin[2])], dtype=np.float64)
    expected = np.asarray([origin + rotation @ point + lift for point in obj], dtype=np.float64)
    rvec, _ = cv2.Rodrigues(_as_matrix(robot_from_camera_pose["R"], (3, 3)))
    tvec = np.asarray(robot_from_camera_pose["t"], dtype=np.float64).reshape(3, 1)
    projected, _ = cv2.projectPoints(
        expected,
        rvec,
        tvec,
        _as_matrix(camera_matrix, (3, 3)),
        np.asarray(dist_coeffs, dtype=np.float64).reshape(-1, 1),
    )
    observed = np.asarray(detected_corners, dtype=np.float64).reshape(-1, 2)
    predicted = np.asarray(projected, dtype=np.float64).reshape(-1, 2)
    if observed.shape != predicted.shape:
        raise ValueError("A validação precisa dos mesmos 54 cantos, em imagens novas")
    pixel_errors = np.linalg.norm(observed - predicted, axis=1)
    back = _backproject_plane(
        observed, camera_matrix, dist_coeffs, robot_from_camera_pose, float(plane_z_mm)
    )
    mm_errors = np.linalg.norm(back[:, :2] - expected[:, :2], axis=1)
    return {
        "plane_z_mm": float(plane_z_mm),
        "pixels": _stats(pixel_errors.tolist()),
        "millimetres": _stats(mm_errors.tolist()),
        "count": int(mm_errors.size),
    }


def _backproject_plane(
    pixels: np.ndarray,
    camera_matrix: list[list[float]],
    dist_coeffs: list[float],
    robot_from_camera_pose: dict[str, Any],
    plane_z_mm: float,
) -> np.ndarray:
    normalized = cv2.undistortPoints(
        np.asarray(pixels, dtype=np.float64).reshape(-1, 1, 2),
        _as_matrix(camera_matrix, (3, 3)),
        np.asarray(dist_coeffs, dtype=np.float64).reshape(-1, 1),
    ).reshape(-1, 2)
    rays = np.concatenate([normalized, np.ones((normalized.shape[0], 1))], axis=1)
    rotation_rc = _as_matrix(robot_from_camera_pose["R"], (3, 3))
    translation_rc = np.asarray(robot_from_camera_pose["t"], dtype=np.float64).reshape(3)
    rotation_cr = rotation_rc.T
    camera_origin = -rotation_cr @ translation_rc
    directions = (rotation_cr @ rays.T).T
    dz = directions[:, 2]
    safe = np.where(np.abs(dz) < 1e-9, 1e-9, dz)
    depth = (float(plane_z_mm) - camera_origin[2]) / safe
    return camera_origin + directions * depth.reshape(-1, 1)


def diagnose_residuals(samples: list[dict[str, float]], xy_tolerance_mm: float) -> dict[str, Any]:
    """Lê o padrão espacial dos resíduos da afim, no formato do guia."""
    if len(samples) < 3:
        return {
            "findings": ["Colete mais pontos do molde para ler o padrão do resíduo."],
            "outlier_count": 0,
            "ransac_available": False,
        }
    errors = np.asarray([sample["error_mm"] for sample in samples], dtype=np.float64)
    radii = np.asarray(
        [
            float(np.hypot(sample["u_norm"] - 0.5, sample["v_norm"] - 0.5))
            for sample in samples
        ],
        dtype=np.float64,
    )
    findings: list[str] = []
    if float(np.std(radii)) > 1e-6 and float(np.std(errors)) > 1e-6:
        correlation = float(np.corrcoef(radii, errors)[0, 1])
        if correlation > 0.6:
            findings.append("O erro cresce com a distância ao centro. Revisar R, t e o referencial.")
    if float(np.std(errors)) < max(2.0, 0.25 * float(np.mean(errors))):
        findings.append("O erro é aproximadamente constante. Validar TCP e a origem física.")
    border = errors[radii > 0.35]
    center = errors[radii < 0.2]
    if border.size and center.size and float(border.mean()) > float(center.mean()) * 1.5 and float(border.mean()) > 5.0:
        findings.append("O erro piora nas bordas. Recalibrar a intrínseca cobrindo melhor o campo.")
    by_plane: dict[float, list[float]] = {}
    for sample in samples:
        by_plane.setdefault(float(sample["z"]), []).append(float(sample["error_mm"]))
    plane_means = [float(np.mean(values)) for values in by_plane.values() if values]
    if len(plane_means) >= 2 and max(plane_means) - min(plane_means) > 5.0:
        findings.append("O erro muda com Z. A paralaxe ainda aparece entre os planos.")
    outlier_count = int(np.sum(errors > float(xy_tolerance_mm)))
    ransac_available = 0 < outlier_count <= 2 and len(samples) - outlier_count >= 3
    if ransac_available:
        findings.append("Poucos pontos muito ruins. O RANSAC pode ser aplicado só nesses outliers.")
    if not findings:
        findings.append("Não há um padrão geométrico dominante nos resíduos desta coleta.")
    return {"findings": findings, "outlier_count": outlier_count, "ransac_available": ransac_available}


def sharpness_score(frame_bgr: np.ndarray, corners: list[list[float]] | None = None) -> float:
    """Variância do Laplaciano num recorte de tamanho fixo. Cai quando o foco abre."""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    crop = gray
    if corners and len(corners) >= 4:
        points = np.asarray(corners, dtype=np.float64).reshape(-1, 2)
        x0, y0 = points.min(axis=0)
        x1, y1 = points.max(axis=0)
        pad = 8
        left = int(max(0, np.floor(x0) - pad))
        top = int(max(0, np.floor(y0) - pad))
        right = int(min(gray.shape[1], np.ceil(x1) + pad))
        bottom = int(min(gray.shape[0], np.ceil(y1) + pad))
        if right - left >= 16 and bottom - top >= 16:
            crop = gray[top:bottom, left:right]
    else:
        height, width = gray.shape
        crop = gray[height // 4 : 3 * height // 4, width // 4 : 3 * width // 4]
    if crop.size < 64:
        return 0.0
    normalized = cv2.resize(crop, (320, 240), interpolation=cv2.INTER_AREA)
    return float(cv2.Laplacian(normalized, cv2.CV_64F).var())


def assess_focus(
    history: list[float],
    peak: float,
    locked_peak: float | None,
) -> dict[str, Any]:
    """Estável no máximo: a janela recente fica perto do pico e pouco oscila."""
    current = float(history[-1]) if history else None
    window = history[-8:]
    new_peak = float(peak)
    if len(history) >= 3:
        smooth = float(np.mean(history[-3:]))
        if smooth > new_peak:
            new_peak = smooth
    stable = False
    if current is not None and len(window) >= 8 and new_peak > 1.0:
        mean = float(np.mean(window))
        deviation = float(np.std(window))
        stable = min(window) >= 0.90 * new_peak and deviation <= 0.08 * max(mean, 1.0)
    soft = bool(locked_peak and current is not None and current < 0.55 * float(locked_peak))
    return {"current": current, "peak": new_peak, "stable": stable, "soft": soft}


def canonical_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def render_checkerboard_frame(
    width: int,
    height: int,
    pose_index: int,
    pattern: tuple[int, int] = PATTERN,
) -> np.ndarray:
    """Tabuleiro sintético em poses variadas, só para o simulador local."""
    square = 28
    squares_x = pattern[0] + 1
    squares_y = pattern[1] + 1
    border = square
    board_w = squares_x * square + 2 * border
    board_h = squares_y * square + 2 * border
    board = np.full((board_h, board_w), 255, np.uint8)
    for row in range(squares_y):
        for col in range(squares_x):
            if (col + row) % 2 == 0:
                x0 = border + col * square
                y0 = border + row * square
                board[y0 : y0 + square, x0 : x0 + square] = 20
    centers = (
        (0.50, 0.50), (0.32, 0.30), (0.68, 0.30), (0.32, 0.70), (0.68, 0.70),
        (0.50, 0.28), (0.30, 0.50), (0.70, 0.50), (0.50, 0.72),
        (0.40, 0.40), (0.60, 0.42), (0.42, 0.62), (0.55, 0.58), (0.38, 0.55),
    )
    cx, cy = centers[int(pose_index) % len(centers)]
    span = min(width, height) * (0.42 + 0.06 * (int(pose_index) % 3))
    aspect = board_w / board_h
    half_w = span * aspect * 0.5
    half_h = span * 0.5
    skew = (int(pose_index) % 5 - 2) * span * 0.04
    center_x = cx * width
    center_y = cy * height
    destination = np.array(
        [
            [center_x - half_w + skew, center_y - half_h],
            [center_x + half_w + skew, center_y - half_h + skew * 0.3],
            [center_x + half_w - skew, center_y + half_h],
            [center_x - half_w - skew, center_y + half_h - skew * 0.2],
        ],
        dtype=np.float32,
    )
    source = np.array(
        [[0, 0], [board_w - 1, 0], [board_w - 1, board_h - 1], [0, board_h - 1]],
        dtype=np.float32,
    )
    homography = cv2.getPerspectiveTransform(source, destination)
    warped = cv2.warpPerspective(board, homography, (width, height), flags=cv2.INTER_LINEAR, borderValue=180)
    return cv2.cvtColor(warped, cv2.COLOR_GRAY2BGR)
