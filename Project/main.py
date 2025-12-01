import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk


class SquatAnalysisApp:
    """
    Minimalstruktur für das Squat-Analyse-Projekt.

    - Öffnet eine Kamera
    - Erkennt ArUco-Marker
    - Berechnet (grob) Femur- und Kniewinkel
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

        # Start-Button
        self.is_measuring = False
        start_button = ttk.Button(self.window, text="Start measurement", command=self.start_measurement)
        start_button.pack(pady=10)

        # Optional: Stop-Button
        stop_button = ttk.Button(self.window, text="Stop measurement", command=self.stop_measurement)
        stop_button.pack(pady=5)

        # --- Kamera Setup ---
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

        # --- ArUco Setup ---
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)

        params = cv2.aruco.DetectorParameters()

        # Etwas großzügigere Threshold-Fenster
        params.adaptiveThreshWinSizeMin = 3
        params.adaptiveThreshWinSizeMax = 23
        params.adaptiveThreshWinSizeStep = 10
        params.adaptiveThreshConstant = 7

        # Markerrand-Anforderungen (wenn Marker kleiner sind, Min-Rate runtersetzen)
        params.minMarkerPerimeterRate = 0.02   # Standard ist höher
        params.maxMarkerPerimeterRate = 4.0

        # Abstand der Ecken – zu klein kann rauschen, zu groß blockiert bei nahen Markern
        params.minCornerDistanceRate = 0.02

        # Ecken genauer verfeinern (macht Erkennung stabiler, aber minimal langsamer)
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

        femur_angle, knee_angle = self.process_frame(frame)

        # GUI-Labels aktualisieren (falls Winkel berechnet werden konnten)
        if femur_angle is not None:
            self.femur_angle_var.set(f"Femur angle: {femur_angle:.1f} °")
        else:
            self.femur_angle_var.set("Femur angle: -- °")

        if knee_angle is not None:
            self.knee_angle_var.set(f"Knee angle: {knee_angle:.1f} °")
        else:
            self.knee_angle_var.set("Knee angle: -- °")

        # Optional: Frame mit eingezeichneten Markern anzeigen
        cv2.imshow("Squat Camera", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.stop_measurement()

        # Nächsten Schritt planen (z.B. alle 20 ms)
        self.window.after(20, self.update_loop)

    # ------------------------------------------------------------------
    # Bildverarbeitung & Winkelberechnung
    # ------------------------------------------------------------------
    def process_frame(self, frame):
        """Detektiert ArUco-Marker im Frame und berechnet die Winkel."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        corners, ids, _ = self.aruco_detector.detectMarkers(gray)

        if ids is not None:
            # Marker im Bild anzeigen
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)

            # in Dict: id -> Mittelpunkt
            centers = {}
            for i, marker_id in enumerate(ids.flatten()):
                pts = corners[i][0]  # 4 Eckpunkte (4x2)
                center = pts.mean(axis=0)  # Mittelwert der x/y-Koordinaten
                centers[int(marker_id)] = center

            femur_angle = self.compute_femur_angle(centers)
            knee_angle = self.compute_knee_angle(centers)
            return femur_angle, knee_angle

        # falls keine Marker erkannt
        return None, None

    def compute_femur_angle(self, centers):
        """
        Femur-Winkel relativ zur Boden-Horizontalen.
        Dafür brauchen wir Hüft- und Knie-Marker.
        """
        if (self.MARKER_HIP_ID not in centers) or (self.MARKER_KNEE_ID not in centers):
            return None

        hip = centers[self.MARKER_HIP_ID]
        knee = centers[self.MARKER_KNEE_ID]

        # Vektor vom Hüftmarker zum Kniewinkel
        vec = knee - hip  # [dx, dy]

        # Bildkoordinaten: y nach unten -> für "mathematischen" Winkel dy invertieren
        dx = vec[0]
        dy = -vec[1]

        angle_rad = np.arctan2(dy, dx)  # Winkel relativ zur x-Achse
        angle_deg = np.degrees(angle_rad)

        # Je nach Definition kannst du hier noch anpassen (z.B. Betrag)
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
