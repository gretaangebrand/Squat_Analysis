from __future__ import annotations # für Python 3.7 Kompatibilität
from dataclasses import dataclass # Typing für Konfiguration
from typing import Dict, Optional # für Typ-Hinweise

import cv2 # OpenCV für ArUco
import numpy as np # NumPy für numerische Operationen

# Markergröße in mm
MARKER_SIZE_MM = 90.0

@dataclass(slots=True) #automatische Initialisierung von Klassen (sammelt Konfigurationsparameter, sonst langes __init__)
# Konfigurationsklasse für ArucoTracker
class ArucoTrackerConfig: # Daten"container" für Konfigurationsparameter
    dictionary: int = cv2.aruco.DICT_6X6_250
    use_corner_refine_subpix: bool = True #stabile Eckenerkennung

    # Toleranzlogik
    update_interval_ms: int = 20   # wie Tkinter-Gui-Loop -> 25 FPS
    max_gap_seconds: float = 0.5   # definierte Toleranz, dass Marker kurz fehlen dürfen -> 0.5 s

    # DetectorParameters (Einstellungen für die Marker-Erkennung)
    adaptiveThreshWinSizeMin: int = 3 # kleinste Fenstergröße von 3x3 Pixeln -> sehr lokale Betrachtung, Marker weit weg, ansonsten würden Marker nicht erkannt werden
    adaptiveThreshWinSizeMax: int = 23 # größte Fenstergröße von 23x23 Pixeln -> große Fenster für nahe Marker und schlechte Lichtverhältnisse
    adaptiveThreshWinSizeStep: int = 10 # OpenCV testet Fenstergrößen in Schritten von 10 Pixeln zwischen min und max -> 3,13,23 -> gute Abdeckung, wenig Rechenaufwand
    adaptiveThreshConstant: int = 7 # Konstante, die von Mittelwert subtrahiert wird -> nur Pixel dunkler als Mittelwert-7 werden als Marker-Pixel betrachtet -> so klare geschlossene Kanten

    minMarkerPerimeterRate: float = 0.02 # Mindestumfang, den Marker haben muss, um erkannt zu werden (im Verhältnis zur Bildgröße = 80px)
    maxMarkerPerimeterRate: float = 4.0 # Höchstumfang, den Marker haben darf, um erkannt zu werden (im Verhältnis zur Bildgröße = 16000px)
    minCornerDistanceRate: float = 0.02 # Mindestabstand zwischen Ecken (im Verhältnis zur Bilddiagonale = 28.28px), dass keine Ecken doppelt erkannt werden/zu nahe beieinander liegen

# ArucoTracker-Klasse
class ArucoTracker:
    """
    Verantwortlich für:
    - detectMarkers()
    - Mittelpunkte berechnen
    - 0.5s Toleranz: Marker dürfen kurz fehlen; danach werden sie verworfen
    """
    # Initialisierung
    def __init__(self, config: Optional[ArucoTrackerConfig] = None):
        # Konfiguration
        self.cfg = config or ArucoTrackerConfig()
        # ArUco-Detector initialisieren
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(self.cfg.dictionary)
        
        # Detector-Parameter setzen
        params = cv2.aruco.DetectorParameters()
        params.adaptiveThreshWinSizeMin = self.cfg.adaptiveThreshWinSizeMin
        params.adaptiveThreshWinSizeMax = self.cfg.adaptiveThreshWinSizeMax
        params.adaptiveThreshWinSizeStep = self.cfg.adaptiveThreshWinSizeStep
        params.adaptiveThreshConstant = self.cfg.adaptiveThreshConstant
 
        # Umfangsrate für Marker-Erkennung definieren
        params.minMarkerPerimeterRate = self.cfg.minMarkerPerimeterRate
        params.maxMarkerPerimeterRate = self.cfg.maxMarkerPerimeterRate
        params.minCornerDistanceRate = self.cfg.minCornerDistanceRate

        # Kalibrierungsdaten, die Skala in mm/px liefern
        self.mm_per_px = None
        self._mm_per_px_ema = None 

        # Ecken verfeinern
        if self.cfg.use_corner_refine_subpix:
            params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX

        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, params)

        # State
        self.frame_counter: int = 0
        self.last_valid_centers: Dict[int, np.ndarray] = {}
        self.last_seen_frame: Dict[int, int] = {}

        # Aus update_interval_ms & max_gap_seconds Frames ableiten
        self.max_gap_frames: int = max(1, int(round(self.cfg.max_gap_seconds * 1000.0 / self.cfg.update_interval_ms)))

    # setzt den Tracker-State zurück, wenn Messung neu startet
    def reset(self) -> None:
        """Setzt den Tracker-State zurück (z.B. bei neuer Messung)."""
        self.frame_counter = 0
        self.last_valid_centers.clear()
        self.last_seen_frame.clear()

    # Frames zählen & Marker erkennen
    def update(self, frame: np.ndarray, draw: bool = True) -> Dict[int, np.ndarray]:
        self.frame_counter += 1

        # Marker in Graustufenbild erkennen
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # pro Marker werden 4 Eckpunkte als Pixel und die IDs zurückgegeben
        corners, ids, _ = self.detector.detectMarkers(gray)

        # Erkannte Marker verarbeiten
        detected_centers: Dict[int, np.ndarray] = {}
        mm_per_px_candidates = []

        # Wenn Marker erkannt wurden, werden deren Mittelpunkte berechnet
        if ids is not None and len(ids) > 0:
            if draw:
                cv2.aruco.drawDetectedMarkers(frame, corners, ids)

            for i, marker_id in enumerate(ids.flatten()):
                pts = corners[i][0]              # (4,2)
                center = pts.mean(axis=0)        # (2,) Durchschnitt der 4 Ecken
                mid = int(marker_id)

                # Mittelpunkt speichern und Frame-Index aktualisieren
                detected_centers[mid] = center
                self.last_valid_centers[mid] = center
                self.last_seen_frame[mid] = self.frame_counter

                # Skala abschätzen
                mmpp = self._estimate_mm_per_px_from_corners(pts)
                if mmpp is not None:
                    mm_per_px_candidates.append(mmpp)

            # robuste Skala: Median + EMA
            if mm_per_px_candidates:
                mm_per_px = float(np.median(mm_per_px_candidates))
                alpha = 0.2
                if self._mm_per_px_ema is None:
                    self._mm_per_px_ema = mm_per_px
                else:
                    self._mm_per_px_ema = (1 - alpha) * self._mm_per_px_ema + alpha * mm_per_px
                self.mm_per_px = self._mm_per_px_ema

        # Toleranz anwenden: nur Marker, die <= max_gap_frames alt sind
        centers_out: Dict[int, np.ndarray] = {}
        for mid, center in self.last_valid_centers.items():
            last_frame = self.last_seen_frame.get(mid, -10**9)
            if (self.frame_counter - last_frame) <= self.max_gap_frames:
                centers_out[mid] = center
        return centers_out

    # Abschätzung mm/px aus sichtbaren Markern
    def _estimate_mm_per_px_from_corners(self, marker_corners_px, marker_size_mm=MARKER_SIZE_MM):
        # marker_corners_px: (4,2)
        c = marker_corners_px
        edges = [
            np.linalg.norm(c[0] - c[1]),
            np.linalg.norm(c[1] - c[2]),
            np.linalg.norm(c[2] - c[3]),
            np.linalg.norm(c[3] - c[0]),
        ]
        mean_edge_px = float(np.mean(edges))
        if mean_edge_px <= 1e-6:
            return None
        return marker_size_mm / mean_edge_px