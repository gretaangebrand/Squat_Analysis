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
    Femur winkel relativ zur Horizontalen (0..90°) mit einem tiefenbasierten Vorzeichen:
    - Betrag/Magnitude: 0° = Femur horizontal, 90° = Femur vertikal
    - Vorzeichen: + wenn Hüfte ÜBER Knie, - wenn Hüfte UNTER Knie
    """
    # Check, ob beide Marker vorhanden sind, sonst keine sinnvolle Berechnung möglich
    if hip_id not in centers or knee_id not in centers:
        return None

    # Marker-Zentren extrahieren
    hip = centers[hip_id]
    knee = centers[knee_id]

    # Vector von Hüfte zu Knie
    vec = knee - hip
    dx = float(vec[0]) # horizontaler Abstand, geht Knie nach links/rechts vom Hip
    dy = float(-vec[1])  # vertikaler Abstand, geht Knie unter/über Hip (invertiert, da Bild-y nach unten wächst)

    # Rohwinkel in Grad relativ zur Horizontalen -> arctan2 zur korrekte Quadrantenbestimmung
    ang = float(np.degrees(np.arctan2(dy, dx)))

    # Steigungswinkel in 0..90° umwandeln --> absolute Winkel wird betrachtet, unabhängig von links/rechts Position von Person
    # wenn Winkel > 90°, dann 180 - Winkel nehmen --> erkennt, dass Person andersherum steht
    mag = abs(ang)
    if mag > 90.0:
        mag = 180.0 - mag

    # wenn Hüfte unter Knie ist, dann negativen Wert zurückgeben
    hip_below_knee = float(hip[1]) > float(knee[1])
    return -mag if hip_below_knee else mag

# Berechnung von Squat-Tiefe: Hüfte relativ zum Knie
def squat_depth_angle_deg(
    centers: Dict[int, np.ndarray],
    hip_id: int,
    knee_id: int,
) -> Optional[float]:
    """
    Squat-Depth-Winkel bezogen auf die "horizontale Linie" durch das Knie.:
    0°  -> Hüfte und Knie auf gleicher Höhe
    <0° -> Hüfte unter Knie (OK)
    >0° -> Hüfte über Knie (zu hoch)

    Definition: angle = arcsin( dy / |femur| ) mit dy = knee_y - hip_y (Bild-y nach unten positiv)
    Wie weit ist die Hüfte unterhalb (negativ) oder oberhalb (positiv) des Knies im Verhältnis zur Femurlänge?
    Femurlänge sorgt dafür, dass unabhängig von KKörpergröße und Abstand Kamera, die Messung vergleichbar ist.
    """
    # Check, ob beide Marker vorhanden sind, sonst keine sinnvolle Berechnung möglich
    if hip_id not in centers or knee_id not in centers:
        return None

    # Marker-Zentren extrahieren
    hip = centers[hip_id]
    knee = centers[knee_id]

    # Femur Vektor berechnen
    vec = knee - hip
    # Femur Länge berechnen
    femur_len = float(np.linalg.norm(vec))
    if femur_len == 0.0:
        return None

    # Vertikaler Abstand dy von Hüfte zu Knie
    dy = float(knee[1] - hip[1])  # Bild-y: nach unten positiv

    # vertikalen Abstand von Hüfe zu Knie durch Femurlänge normieren -> Größenunabhängigkeit
    # und Messwert begrenzen auf [-1, 1] für arcsin
    ratio = np.clip(dy / femur_len, -1.0, 1.0)

    # arcsin von normiertem Wert berechnen und in Grad umwandeln
    angle_rad = np.arcsin(ratio)
    return float(np.degrees(angle_rad))

# Berechnung des anatomischen Kniegelenkswinkel (Innenwinkel) zwischen Femur und Tibia
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
    # Check, ob alle drei Marker vorhanden sind, sonst keine sinnvolle Berechnung möglich
    if hip_id not in centers or knee_id not in centers or ankle_id not in centers:
        return None

    # Marker-Zentren extrahieren
    hip = centers[hip_id]
    knee = centers[knee_id]
    ankle = centers[ankle_id]

    # Vektoren berechnen
    v1 = hip - knee
    v2 = ankle - knee

    # Winkel zwischen Vektoren berechnen
    norm_prod = float(np.linalg.norm(v1) * np.linalg.norm(v2))
    if norm_prod == 0.0:
        return None

    # Winkel berechnen via arccos(dot(v1,v2) / (|v1|*|v2|))
    cos_angle = np.clip(float(np.dot(v1, v2)) / norm_prod, -1.0, 1.0)
    angle_rad = np.arccos(cos_angle)
    return float(np.degrees(angle_rad))

# Berechnung des "validen" Kniewinkels für gültige Squat-Analyse
# Der valide Winkel ist definiert als Abweichung vom anatomischen Referenzwinkel um 90°
def knee_valid_angle_deg(centers, hip_id, knee_id, ankle_id, reference_deg=90.0):
    """
    Valid/functional knee angle:
    0° at reference_deg (default 90° anatomical),
    positive angle = higher squat (more extended),
    negative angle = deeper squat (more flexed).
    """
    # Umrechnung von anatomischen Winkel in validen Winkel
    anat = knee_angle_deg(centers, hip_id, knee_id, ankle_id)
    if anat is None:
        return None
    return anat - reference_deg
