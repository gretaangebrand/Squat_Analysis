import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk


class SquatAnalysisApp:
    """
    Minimalstruktur für das Squat-Analyse-Projekt.

    - Öffnet eine Kamera
    - Erkennt ArUco-Marker
    - Berechnet Femur- und Kniewinkel
    - Zeigt die Winkel in einer Tkinter-GUI an
    """

    # Marker anpassen je nach Platzierung
    MARKER_HIP_ID = 42     # Marker z.B. am Hüftknochen/Femur-Proximal
    MARKER_KNEE_ID = 41    # Marker am Knie
    MARKER_ANKLE_ID = 40   # Marker am Knöchel/Unterschenkel Distal

    def __init__(self, camera_index: int = 1):
        # --- GUI Setup ---
        self.window = tk.Tk()
        self.window.title("Squat Analysis – Minimal Version")

        # Labels für Winkel
        self.femur_angle_var = tk.StringVar(value="Femur angle: -- °")
        self.knee_angle_var = tk.StringVar(value="Knee angle: -- °")

        femur_label = ttk.Label(self.window, textvariable=self.femur_angle_var, font=("Arial", 16))
        knee_label = ttk.Label(self.window, textvariable=self.knee_angle_var, font=("Arial", 16))

        femur_label.pack(pady=10)
        knee_label.pack(pady=10)

        # Squat approval label
        self.squat_status_var = tk.StringVar(value="Squat: --")
        squat_label = ttk.Label(self.window, textvariable=self.squat_status_var, font=("Arial", 16))
        squat_label.pack(pady=10)

        # Start-Button
        self.is_measuring = False
        start_button = ttk.Button(self.window, text="Start measurement", command=self.start_measurement)
        start_button.pack(pady=10)

        # Optional: Stop-Button
        stop_button = ttk.Button(self.window, text="Stop measurement", command=self.stop_measurement)
        stop_button.pack(pady=5)

        # --- Kamera Setup ---
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            print("❌ Cannot open camera")
        else:
            print("Camera opened successfully")

        # moderate Auflösung für weniger Rechenaufwand
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # feste GUI-Update-Rate: 40 ms ≈ 25 fps
        self.update_interval = 40
        print(f"GUI update interval (ms): {self.update_interval}")

        # Letzte Marker-Zentren speichern (für Visualisierung)
        self.last_centers = {}

        # --- ArUco Setup ---
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)

        params = cv2.aruco.DetectorParameters()

        # Etwas großzügigere Threshold-Fenster
        params.adaptiveThreshWinSizeMin = 3
        params.adaptiveThreshWinSizeMax = 23
        params.adaptiveThreshWinSizeStep = 10
        params.adaptiveThreshConstant = 7

        # Markerrand-Anforderungen (wenn Marker kleiner sind, Min-Rate runtersetzen)
        params.minMarkerPerimeterRate = 0.02
        params.maxMarkerPerimeterRate = 4.0

        # Abstand der Ecken
        params.minCornerDistanceRate = 0.02

        # Ecken genauer verfeinern
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX

        self.aruco_params = params
        self.aruco_detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)

        # Zum sauberen Beenden
        self.window.protocol("WM_DELETE_WINDOW", self.on_close)

    # ------------------------------------------------------------------
    # GUI Steuerung
    # ------------------------------------------------------------------
    def start_measurement(self):
        """Wird beim Klick auf den Start-Button aufgerufen."""
        if not self.is_measuring:
            self.is_measuring = True
            self.update_loop()  # ersten Durchlauf starten
            print("Measurement started")

    def stop_measurement(self):
        """Messung stoppen."""
        self.is_measuring = False
        print("Measurement stopped")

    def on_close(self):
        """Fenster schließen -> Ressourcen freigeben."""
        self.is_measuring = False
        if self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()
        self.window.destroy()

    def run(self):
        """Tkinter-Hauptloop starten."""
        self.window.mainloop()

    # ------------------------------------------------------------------
    # Hauptprozess: 1) Frame holen 2) Marker finden 3) Winkel 4) GUI updaten
    # ------------------------------------------------------------------
    def update_loop(self):
        """Ein Schritt des Messloops. Wird über window.after() regelmäßig aufgerufen."""
        if not self.is_measuring:
            return  # nichts tun, wenn Messung pausiert ist

        ret, frame = self.cap.read()
        if not ret:
            print("Failed to grab frame")
            # in 100 ms nochmal versuchen
            self.window.after(100, self.update_loop)
            return

        femur_angle, knee_angle, squat_depth_angle = self.process_frame(frame)

        # Anatomischen Femur-Segmentwinkel anzeigen
        if femur_angle is not None:
            self.femur_angle_var.set(f"Femur angle: {femur_angle:.1f} °")
        else:
            self.femur_angle_var.set("Femur angle: -- °")

        # Knie-Winkel anzeigen
        if knee_angle is not None:
            self.knee_angle_var.set(f"Knee angle: {knee_angle:.1f} °")
        else:
            self.knee_angle_var.set("Knee angle: -- °")

        # Squat approval basierend auf Squat-Depth-Winkel
        if squat_depth_angle is not None:
            if squat_depth_angle <= 0:   # 0° und tiefer = OK
                self.squat_status_var.set("Squat: ✅ depth OK")
            else:
                self.squat_status_var.set("Squat: ❌ too high")
        else:
            self.squat_status_var.set("Squat: --")

        # --- Boden-Referenzlinie ---
        height, width, _ = frame.shape
        y_floor = int(height * 0.8)
        cv2.line(frame, (0, y_floor), (width, y_floor), (0, 255, 0), 2)

        # --- Segmente einzeichnen, falls Marker da ---
        has_hip = self.MARKER_HIP_ID in self.last_centers
        has_knee = self.MARKER_KNEE_ID in self.last_centers
        has_ankle = self.MARKER_ANKLE_ID in self.last_centers

        if has_hip:
            hip = self.last_centers[self.MARKER_HIP_ID]
            hip_pt = (int(hip[0]), int(hip[1]))
            cv2.circle(frame, hip_pt, 6, (0, 0, 255), -1)   # Hüfte = rot

        if has_knee:
            knee = self.last_centers[self.MARKER_KNEE_ID]
            knee_pt = (int(knee[0]), int(knee[1]))
            cv2.circle(frame, knee_pt, 6, (255, 0, 0), -1)  # Knie = blau

        if has_ankle:
            ankle = self.last_centers[self.MARKER_ANKLE_ID]
            ankle_pt = (int(ankle[0]), int(ankle[1]))
            cv2.circle(frame, ankle_pt, 6, (0, 255, 0), -1)  # Knöchel = grün

        # Oberschenkel (Hüfte -> Knie) = rote Linie
        if has_hip and has_knee:
            cv2.line(frame, hip_pt, knee_pt, (0, 0, 255), 2)

        # Unterschenkel (Knie -> Knöchel) = grüne Linie
        if has_knee and has_ankle:
            cv2.line(frame, knee_pt, ankle_pt, (0, 255, 0), 2)


        # Frame mit eingezeichneten Markern anzeigen
        cv2.imshow("Squat Camera", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.stop_measurement()

        # Nächsten Schritt planen
        self.window.after(self.update_interval, self.update_loop)

    # ------------------------------------------------------------------
    # Bildverarbeitung & Winkelberechnung
    # ------------------------------------------------------------------
    def process_frame(self, frame):
        """Detektiert ArUco-Marker im Frame und berechnet die Winkel."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        corners, ids, _ = self.aruco_detector.detectMarkers(gray)

        if ids is not None:
            # Marker anzeigen
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)

            # Neue sichtbare Marker sammeln
            centers = {}
            for i, marker_id in enumerate(ids.flatten()):
                pts = corners[i][0]           # 4 Eckpunkte
                center = pts.mean(axis=0)     # Mittelpunkt
                centers[int(marker_id)] = center

            # Sichtbare Marker in last_valid_centers aktualisieren
            for mid, c in centers.items():
                self.last_valid_centers[mid] = c

            # Nutze alle bekannten Marker (sichtbar + zuletzt gültig)
            all_centers = self.last_valid_centers.copy()

        else:
            # Keine Marker erkannt → fallback auf last_valid_centers
            all_centers = self.last_valid_centers.copy()

        # Für Visualisierung speichern
        self.last_centers = all_centers

        # Winkel berechnen
        femur_segment_angle = self.compute_femur_segment_angle(all_centers)
        squat_depth_angle = self.compute_squat_depth_angle(all_centers)
        knee_angle = self.compute_knee_angle(all_centers)

        return femur_segment_angle, knee_angle, squat_depth_angle


    def compute_femur_segment_angle(self, centers):
        """
        Anatomischer Femur-Segmentwinkel relativ zur Horizontalen.
        0°  -> Femur horizontal
        +90°/-90° -> vertikal
        """
        if (self.MARKER_HIP_ID not in centers) or (self.MARKER_KNEE_ID not in centers):
            return None

        hip = centers[self.MARKER_HIP_ID]
        knee = centers[self.MARKER_KNEE_ID]

        vec = knee - hip  # [dx, dy]
        dx = vec[0]
        dy = -vec[1]  # Bild-y nach oben drehen

        angle_rad = np.arctan2(dy, dx)
        angle_deg = np.degrees(angle_rad)
        return angle_deg

    def compute_squat_depth_angle(self, centers):
        """
        Squat-Depth-Winkel:
        0°  -> Hüfte und Knie auf gleicher Höhe (Femur waagrecht)
        <0° -> Hüfte unter Knie (guter, tiefer Squat)
        >0° -> Hüfte über Knie (zu hoch)
        """
        if (self.MARKER_HIP_ID not in centers) or (self.MARKER_KNEE_ID not in centers):
            return None

        hip = centers[self.MARKER_HIP_ID]   # [x_hip, y_hip]
        knee = centers[self.MARKER_KNEE_ID] # [x_knee, y_knee]

        vec = knee - hip
        femur_len = np.linalg.norm(vec)
        if femur_len == 0:
            return None

        # y-Koordinate: nach unten positiv
        dy = knee[1] - hip[1]  # >0: Knie tiefer als Hüfte (Hüfte höher); <0 umgekehrt

        ratio = np.clip(dy / femur_len, -1.0, 1.0)
        angle_rad = np.arcsin(ratio)
        angle_deg = np.degrees(angle_rad)

        return angle_deg

    def compute_knee_angle(self, centers):
        """
        Knie-Winkel als Winkel zwischen Femur- und Unterschenkelvektor.
        Wir nehmen den Winkel zwischen
        - V1: Knie -> Hüfte
        - V2: Knie -> Knöchel
        """
        if (self.MARKER_HIP_ID not in centers or
                self.MARKER_KNEE_ID not in centers or
                self.MARKER_ANKLE_ID not in centers):
            return None

        hip = centers[self.MARKER_HIP_ID]
        knee = centers[self.MARKER_KNEE_ID]
        ankle = centers[self.MARKER_ANKLE_ID]

        v1 = hip - knee    # Femur-Richtung
        v2 = ankle - knee  # Unterschenkel-Richtung

        # Kosinus des Winkels
        dot = float(np.dot(v1, v2))
        norm_prod = np.linalg.norm(v1) * np.linalg.norm(v2)
        if norm_prod == 0:
            return None

        cos_angle = np.clip(dot / norm_prod, -1.0, 1.0)
        angle_rad = np.arccos(cos_angle)
        angle_deg = np.degrees(angle_rad)
        return angle_deg


if __name__ == "__main__":
    app = SquatAnalysisApp(camera_index=1)
    app.run()
