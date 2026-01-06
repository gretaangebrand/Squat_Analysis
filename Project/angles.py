# kinematics/angles.py
from __future__ import annotations # for forward references
from typing import Dict, Optional # for type hints
import numpy as np # for numerical operations

# Anatomische Winkelberechnungen basierend auf ArUco-Marker-Zentren 
# und relativ zur Horizontalen Bild-x-Achse
def femur_angle_depth_signed_deg(
centers: Dict[int, np.ndarray],
    hip_id: int,
    knee_id: int,
) -> Optional[float]:
    """
    Femur angle relative to horizontal (0..90°) with a depth-based sign:
    - magnitude: 0° = femur horizontal, 90° = femur vertical
    - sign: + if hip is ABOVE knee, - if hip is BELOW knee
    """
    # Check, ob beide Marker vorhanden sind, sonst keine sinnvolle Berechnung möglich
    if hip_id not in centers or knee_id not in centers:
        return None

    hip = centers[hip_id]
    knee = centers[knee_id]

    # Vector hip -> knee (direction doesn't matter for magnitude)
    vec = knee - hip
    dx = float(vec[0])
    dy = float(-vec[1])  # invert image-y for math-consistent angle

    # raw angle relative to +x in degrees (-180..180)
    ang = float(np.degrees(np.arctan2(dy, dx)))

    # magnitude relative to horizontal in [0..90]
    # (take absolute, then fold angles >90 back into 0..90)
    mag = abs(ang)
    if mag > 90.0:
        mag = 180.0 - mag

    # depth-based sign: image y grows downward
    hip_below_knee = float(hip[1]) > float(knee[1])
    return -mag if hip_below_knee else mag


# Berechnung von Squat-Tiefe: Hüfte relativ zum Knie
def squat_depth_angle_deg(
    centers: Dict[int, np.ndarray],
    hip_id: int,
    knee_id: int,
) -> Optional[float]:
    """
    Squat-Depth-Winkel bezogen auf die horizontale Linie durch das Knie.:
    0°  -> Hüfte und Knie auf gleicher Höhe
    <0° -> Hüfte unter Knie (OK)
    >0° -> Hüfte über Knie (zu hoch)

    Definition: angle = arcsin( dy / |femur| ) mit dy = knee_y - hip_y (Bild-y nach unten positiv)
    """
    if hip_id not in centers or knee_id not in centers:
        return None

    hip = centers[hip_id]
    knee = centers[knee_id]

    # Femur Vektor und Länge
    vec = knee - hip
    femur_len = float(np.linalg.norm(vec))
    if femur_len == 0.0:
        return None

    # Vertikaler Abstand dy von Hüfte zu Knie
    dy = float(knee[1] - hip[1])  # Bild-y: nach unten positiv
    # vertikalen Abstand durch Femurlänge normieren --> Größenunabhängigkeit
    # und Messwert begrenzen auf [-1, 1] für arcsin
    ratio = np.clip(dy / femur_len, -1.0, 1.0)

    angle_rad = np.arcsin(ratio)
    return float(np.degrees(angle_rad))

# Berechnung des anatomischen Kniewinkels (Innenwinkel) zwischen Femur und Tibia
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

# Berechnung des "validen" Kniewinkels für gültige Squat-Analyse
def knee_valid_angle_deg(centers, hip_id, knee_id, ankle_id, reference_deg=90.0):
    """
    Valid/functional knee angle:
    0° at reference_deg (default 90° anatomical),
    positive = higher squat (more extended),
    negative = deeper squat (more flexed).
    """
    # Umrechnung von anatomischen Winkel in validen Winkel
    anat = knee_angle_deg(centers, hip_id, knee_id, ankle_id)
    if anat is None:
        return None
    return anat - reference_deg
