from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

MARKER_SIZE_MM = 90.0

@dataclass
class ArucoTrackerConfig:
    dictionary: int = cv2.aruco.DICT_6X6_250
    use_corner_refine_subpix: bool = True

    # Toleranzlogik
    update_interval_ms: int = 40   # entspricht deinem GUI-Tick
    max_gap_seconds: float = 1.0   # gewünschte Toleranz (1 s)

    # DetectorParameters (deine aktuellen Settings)
    adaptiveThreshWinSizeMin: int = 3
    adaptiveThreshWinSizeMax: int = 23
    adaptiveThreshWinSizeStep: int = 10
    adaptiveThreshConstant: int = 7

    minMarkerPerimeterRate: float = 0.02
    maxMarkerPerimeterRate: float = 4.0
    minCornerDistanceRate: float = 0.02


class ArucoTracker:
    """
    Verantwortlich für:
    - detectMarkers()
    - Mittelpunkte berechnen
    - 1s Toleranz: Marker dürfen kurz fehlen; danach werden sie verworfen
    """

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
 
        params.minMarkerPerimeterRate = self.cfg.minMarkerPerimeterRate
        params.maxMarkerPerimeterRate = self.cfg.maxMarkerPerimeterRate
        params.minCornerDistanceRate = self.cfg.minCornerDistanceRate

        # Kalibrierungsdaten
        self.mm_per_px = None
        self._mm_per_px_ema = None  # geglättet

        # Corner-Refinement
        if self.cfg.use_corner_refine_subpix:
            params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX

        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, params)

        # State
        self.frame_counter: int = 0
        self.last_valid_centers: Dict[int, np.ndarray] = {}
        self.last_seen_frame: Dict[int, int] = {}

        # Aus update_interval_ms & max_gap_seconds Frames ableiten
        self.max_gap_frames: int = max(
            1, int(round(self.cfg.max_gap_seconds * 1000.0 / self.cfg.update_interval_ms))
        )

    def reset(self) -> None:
        """Setzt den Tracker-State zurück (z.B. bei neuer Messung)."""
        self.frame_counter = 0
        self.last_valid_centers.clear()
        self.last_seen_frame.clear()

    def update(self, frame: np.ndarray, draw: bool = True) -> Dict[int, np.ndarray]:
        """
        Nimmt ein BGR-Frame und liefert centers zurück:
        {marker_id: np.array([x, y])}
        """
        self.frame_counter += 1

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self.detector.detectMarkers(gray)

        if ids is not None and len(ids) > 0:
            if draw:
                cv2.aruco.drawDetectedMarkers(frame, corners, ids)

            # sichtbare Marker → Mittelpunkte + State Update
            for i, marker_id in enumerate(ids.flatten()):
                pts = corners[i][0]              # (4,2)
                center = pts.mean(axis=0)        # (2,)
                mid = int(marker_id)

                self.last_valid_centers[mid] = center
                self.last_seen_frame[mid] = self.frame_counter

            mm_per_px_candidates = []

            for i, marker_id in enumerate(ids.flatten()):
                pts = corners[i][0]  # 4x2
                center = pts.mean(axis=0)
                centers[int(marker_id)] = center

                mmpp = self._estimate_mm_per_px_from_corners(pts)
                if mmpp is not None:
                    mm_per_px_candidates.append(mmpp)

            # robuste Skala: Median (weniger empfindlich auf Ausreißer)
            if mm_per_px_candidates:
                mm_per_px = float(np.median(mm_per_px_candidates))

                # Glättung (EMA) damit es nicht flackert
                alpha = 0.2  # 0.1..0.3 sinnvoll
                if self._mm_per_px_ema is None:
                    self._mm_per_px_ema = mm_per_px
                else:
                    self._mm_per_px_ema = (1 - alpha) * self._mm_per_px_ema + alpha * mm_per_px

                self.mm_per_px = self._mm_per_px_ema


        # Toleranz anwenden: nur Marker, die <= max_gap_frames "alt" sind
        centers: Dict[int, np.ndarray] = {}
        for mid, center in self.last_valid_centers.items():
            last_frame = self.last_seen_frame.get(mid, -10**9)
            if (self.frame_counter - last_frame) <= self.max_gap_frames:
                centers[mid] = center
        return centers

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
