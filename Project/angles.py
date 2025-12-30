# kinematics/angles.py
from __future__ import annotations

from typing import Dict, Optional

import numpy as np


def femur_segment_angle_deg(
    centers: Dict[int, np.ndarray],
    hip_id: int,
    knee_id: int,
) -> Optional[float]:
    """
    Anatomischer Femur-Segmentwinkel relativ zur Horizontalen (Bild-x-Achse).
    0°  -> Femur horizontal
    +90°/-90° -> vertikal

    Hinweis: Bildkoordinaten haben y nach unten. Für "mathematischen" Winkel wird dy invertiert.
    """
    if hip_id not in centers or knee_id not in centers:
        return None

    hip = centers[hip_id]
    knee = centers[knee_id]

    vec = knee - hip
    dx = float(vec[0])
    dy = float(-vec[1])  # invertiere Bild-y

    angle_rad = np.arctan2(dy, dx)
    return float(np.degrees(angle_rad))


def squat_depth_angle_deg(
    centers: Dict[int, np.ndarray],
    hip_id: int,
    knee_id: int,
) -> Optional[float]:
    """
    Squat-Depth-Winkel (wie in deiner Grafik):
    0°  -> Hüfte und Knie auf gleicher Höhe
    <0° -> Hüfte unter Knie (OK)
    >0° -> Hüfte über Knie (zu hoch)

    Definition: angle = arcsin( dy / |femur| ) mit dy = knee_y - hip_y (Bild-y nach unten positiv)
    """
    if hip_id not in centers or knee_id not in centers:
        return None

    hip = centers[hip_id]
    knee = centers[knee_id]

    vec = knee - hip
    femur_len = float(np.linalg.norm(vec))
    if femur_len == 0.0:
        return None

    dy = float(knee[1] - hip[1])  # Bild-y: nach unten positiv
    ratio = np.clip(dy / femur_len, -1.0, 1.0)

    angle_rad = np.arcsin(ratio)
    return float(np.degrees(angle_rad))


def knee_angle_deg(
    centers: Dict[int, np.ndarray],
    hip_id: int,
    knee_id: int,
    ankle_id: int,
) -> Optional[float]:
    """
    Kniewinkel (Innenwinkel) zwischen:
    v1 = hip - knee
    v2 = ankle - knee
    """
    if hip_id not in centers or knee_id not in centers or ankle_id not in centers:
        return None

    hip = centers[hip_id]
    knee = centers[knee_id]
    ankle = centers[ankle_id]

    v1 = hip - knee
    v2 = ankle - knee

    norm_prod = float(np.linalg.norm(v1) * np.linalg.norm(v2))
    if norm_prod == 0.0:
        return None

    cos_angle = np.clip(float(np.dot(v1, v2)) / norm_prod, -1.0, 1.0)
    angle_rad = np.arccos(cos_angle)
    return float(np.degrees(angle_rad))


def knee_valid_angle_deg(centers, hip_id, knee_id, ankle_id, reference_deg=90.0):
    """
    Valid/functional knee angle:
    0° at reference_deg (default 90° anatomical),
    positive = higher squat (more extended),
    negative = deeper squat (more flexed).
    """
    anat = knee_angle_deg(centers, hip_id, knee_id, ankle_id)
    if anat is None:
        return None
    return anat - reference_deg


def update_depth_stability(self, depth_angle):
    """
    Updatet Streaks und liefert:
    - depth_class: "OK" / "HIGH" / "MID" / None
    - ok_stable: True wenn OK lange genug
    - high_stable: True wenn HIGH lange genug
    """
    if depth_angle is None:
        self._ok_streak = 0
        self._high_streak = 0
        return None, False, False

    # Klassifikation mit Hysterese-Band
    if depth_angle <= self.depth_ok_threshold:
        depth_class = "OK"
        self._ok_streak += 1
        self._high_streak = 0
    elif depth_angle >= self.depth_high_threshold:
        depth_class = "HIGH"
        self._high_streak += 1
        self._ok_streak = 0
    else:
        depth_class = "MID"
        # in MID zählen wir keine streak hoch, sondern resetten (stabiler)
        self._ok_streak = 0
        self._high_streak = 0

    ok_stable = self._ok_streak >= self.stable_frames_required
    high_stable = self._high_streak >= self.stable_frames_required
    return depth_class, ok_stable, high_stable
