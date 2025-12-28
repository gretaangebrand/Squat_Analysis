import cv2
import tkinter as tk
from tkinter import ttk
import sys
import winsound

from aruco_tracker import ArucoTracker, ArucoTrackerConfig
from angles import femur_segment_angle_deg, squat_depth_angle_deg, knee_angle_deg

from collections import deque
import matplotlib
matplotlib.use("TkAgg")

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# =========================
# Plot configuration
# =========================
WINDOW_SECONDS = 6
FPS_ESTIMATE = 25
MAX_SAMPLES = WINDOW_SECONDS * FPS_ESTIMATE  # 150

class SquatAnalysisApp:
    """
    Squat Analysis – Main GUI App

    - Öffnet Kamera
    - Holt Markerzentren über ArucoTracker (ausgelagert)
    - Berechnet Winkel
    - Zeichnet Segmente + zeigt Werte in GUI
    - Tracking Status + stabiler Valid-Squat Counter (State Machine)
    """

    # Marker IDs anpassen je nach Platzierung
    MARKER_HIP_ID = 42
    MARKER_KNEE_ID = 41
    MARKER_ANKLE_ID = 40
    MARKER_BAR_ID = 43   # Bar/Handle marker


    def __init__(self, camera_index: int = 1):
        # -------------------------
        # GUI Setup
        # -------------------------
        self.window = tk.Tk()
        self.window.title("Squat Analysis")

        self.femur_angle_var = tk.StringVar(value="Femur angle: -- °")
        self.knee_angle_var = tk.StringVar(value="Knee angle: -- °")
        self.squat_status_var = tk.StringVar(value="Squat: --")
        self.tracking_status_var = tk.StringVar(value="Tracking: --")
        # -------------------------
        # Handle / Bar height tracking
        # -------------------------
        self.bar_y_ref = None              # Referenz-y (Stand) in Pixel
        self.bar_height_px = None          # aktuelle Höhe relativ zur Referenz (Pixel)

        self.bar_height_var = tk.StringVar(value="Bar height: -- px")
        ttk.Label(self.window, textvariable=self.bar_height_var, font=("Arial", 16)).pack(pady=10)


        ttk.Label(self.window, textvariable=self.femur_angle_var, font=("Arial", 16)).pack(pady=10)
        ttk.Label(self.window, textvariable=self.knee_angle_var, font=("Arial", 16)).pack(pady=10)
        ttk.Label(self.window, textvariable=self.squat_status_var, font=("Arial", 16)).pack(pady=10)
        ttk.Label(self.window, textvariable=self.tracking_status_var, font=("Arial", 14)).pack(pady=6)

        self.is_measuring = False
        ttk.Button(self.window, text="Start measurement", command=self.start_measurement).pack(pady=10)
        ttk.Button(self.window, text="Stop measurement", command=self.stop_measurement).pack(pady=5)

        # -------------------------
        # Squat Counter / State Machine
        # -------------------------
        self.rep_count = 0
        self.rep_count_var = tk.StringVar(value="Valid squats: 0")
        ttk.Label(self.window, textvariable=self.rep_count_var, font=("Arial", 16)).pack(pady=10)
        # --- Sound on valid squat (checkbox) ---
        self.sound_enabled_var = tk.BooleanVar(value=True)  # default an (oder False, wie du willst)
        ttk.Checkbutton(self.window, text="Sound on valid squat", variable=self.sound_enabled_var).pack(pady=6)


        # Hysterese (gegen Flackern)
        self.depth_ok_threshold = -2.0     # <= -> OK (tief genug)
        self.depth_high_threshold = 2.0   # >= -> HIGH (zu hoch)

        # Mindestdauer (Frames) für stabile Zustände
        self.stable_frames_required = 5

        # interne Zähler
        self._ok_streak = 0
        self._high_streak = 0

        # Zustände: "READY" (oben), "BOTTOM" (unten stabil erreicht)
        self._squat_state = "READY"

        # -------------------------
        # Kamera Setup
        # -------------------------
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

        # -------------------------
        # ArUco Tracker (ausgelagert)
        # -------------------------
        tracker_cfg = ArucoTrackerConfig(
            dictionary=cv2.aruco.DICT_6X6_250,
            update_interval_ms=self.update_interval,
            max_gap_seconds=1.0,  # 1s Toleranz für verdeckte Marker
        )
        self.tracker = ArucoTracker(tracker_cfg)

        # Letzte Marker-Zentren (für Zeichnung)
        self.last_centers = {}

        # -------------------------
        # Knee angle buffer (last X seconds)
        # -------------------------
        self.knee_angle_buffer = deque(maxlen=150)  # 6 Sekunden @ ~25 fps
        # -------------------------
        # Matplotlib plot (embedded)
        # -------------------------
        self.fig = Figure(figsize=(6, 2.5), dpi=100)
        self.ax = self.fig.add_subplot(111)

        self.ax.set_title("Knee Angle (last 6 seconds)")
        self.ax.set_xlabel("Time (frames)")
        self.ax.set_ylabel("Angle (deg)")
        self.ax.set_ylim(0, 180)
        self.ax.set_xlim(0, 150)
        self.ax.grid(True)

        self.knee_line, = self.ax.plot([], [], color="blue", linewidth=2)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.window)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, pady=10)


        # sauberes Beenden
        self.window.protocol("WM_DELETE_WINDOW", self.on_close)

    # ------------------------------------------------------------------
    # GUI Steuerung
    # ------------------------------------------------------------------
    def start_measurement(self):
        if not self.is_measuring:
            self.tracker.reset()
            self.bar_y_ref = None
            self.bar_height_px = None
            self.bar_height_var.set("Bar height: -- px")


            # counter/state reset (optional aber sinnvoll)
            self.rep_count = 0
            self.rep_count_var.set("Valid squats: 0")
            self._squat_state = "READY"
            self._ok_streak = 0
            self._high_streak = 0

            self.is_measuring = True
            self.update_loop()
            print("Measurement started")

    def stop_measurement(self):
        self.is_measuring = False
        print("Measurement stopped")

    def on_close(self):
        self.is_measuring = False
        if self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()
        self.window.destroy()

    def run(self):
        self.window.mainloop()

    # ------------------------------------------------------------------
    # Hauptloop
    # ------------------------------------------------------------------
    def update_loop(self):
        if not self.is_measuring:
            return

        try:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to grab frame")
                self.window.after(100, self.update_loop)
                return

            femur_angle, knee_angle, squat_depth_angle, bar_height_px = self.process_frame(frame)

            # Tracking Status updaten (basierend auf last_centers)
            self.update_tracking_status(self.last_centers)

            # GUI Winkel
            if femur_angle is not None:
                self.femur_angle_var.set(f"Femur angle: {femur_angle:.1f} °")
            else:
                self.femur_angle_var.set("Femur angle: -- °")

            if knee_angle is not None:
                self.knee_angle_var.set(f"Knee angle: {knee_angle:.1f} °")
            else:
                self.knee_angle_var.set("Knee angle: -- °")
            # --- Knee angle buffer update ---
            if knee_angle is not None:
                self.knee_angle_buffer.append(knee_angle)
            else:
                # Optional: Lücke (verhindert Zacken)
                self.knee_angle_buffer.append(float("nan"))
            # --- Bar height GUI update ---
            if bar_height_px is not None:
                self.bar_height_var.set(f"Bar height: {bar_height_px:.1f} px")
            else:
                self.bar_height_var.set("Bar height: -- px")


            # --- Update knee angle plot ---
            y_data = list(self.knee_angle_buffer)
            x_data = list(range(len(y_data)))

            self.knee_line.set_data(x_data, y_data)
            self.ax.set_xlim(0, max(150, len(y_data)))
            self.canvas.draw_idle()


            # --- Stabilisierung + Statusanzeige + Counter ---
            depth_class, ok_stable, high_stable = self.update_depth_stability(squat_depth_angle)

            # Anzeige (mit borderline)
            if depth_class is None:
                self.squat_status_var.set("Squat: --")
            elif depth_class == "OK":
                self.squat_status_var.set("Squat: ✅ depth OK")
            elif depth_class == "HIGH":
                self.squat_status_var.set("Squat: ❌ too high")
            else:
                self.squat_status_var.set("Squat: ⚠ borderline")

            # Wenn kein Depth-Signal (z.B. Marker fehlen), State zurücksetzen
            if depth_class is None:
                self._squat_state = "READY"

            # State Machine fürs Zählen:
            # READY -> BOTTOM wenn unten stabil erreicht
            if self._squat_state == "READY":
                if ok_stable:
                    self._squat_state = "BOTTOM"

            # BOTTOM -> READY (+1) wenn wieder oben stabil erreicht
            elif self._squat_state == "BOTTOM":
                if high_stable:
                    self.rep_count += 1
                    self.rep_count_var.set(f"Valid squats: {self.rep_count}")

                    # ✅ SOUND TRIGGER
                    self.play_valid_squat_sound()

                    self._squat_state = "READY"


            # Boden-Referenzlinie
            height, width, _ = frame.shape
            y_floor = int(height * 0.8)
            cv2.line(frame, (0, y_floor), (width, y_floor), (0, 255, 0), 2)

            # Segmente zeichnen
            self.draw_segments(frame)

            # Overlay im Video (optional hilfreich)
            cv2.putText(frame, f"Reps: {self.rep_count}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            cv2.putText(frame, f"State: {self._squat_state}", (10, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            # Frame anzeigen
            cv2.imshow("Squat Camera", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                self.stop_measurement()

            self.window.after(self.update_interval, self.update_loop)

        except Exception as e:
            # robustes Verhalten: stop + anzeigen
            print(f"❌ Error in update_loop: {e}")
            self.squat_status_var.set("Squat: ERROR (see console)")
            self.stop_measurement()

    # ------------------------------------------------------------------
    # Prozess: Marker -> Winkel
    # ------------------------------------------------------------------
    def process_frame(self, frame):
        centers = self.tracker.update(frame, draw=True)
        self.last_centers = centers

        # --- Bar height computation (relative to start position) ---
        if self.MARKER_BAR_ID in centers:
            bar_y = float(centers[self.MARKER_BAR_ID][1])

            # Referenz beim ersten gültigen Frame setzen
            if self.bar_y_ref is None:
                self.bar_y_ref = bar_y

            self.bar_height_px = bar_y - self.bar_y_ref  # >0 wenn bar nach unten (tiefer = größerer y-Wert)
        else:
            # wenn BAR fehlt: lasse letzten Wert stehen (oder setze None, je nachdem was du willst)
            self.bar_height_px = self.bar_height_px


        femur_angle = femur_segment_angle_deg(centers, self.MARKER_HIP_ID, self.MARKER_KNEE_ID)
        depth_angle = squat_depth_angle_deg(centers, self.MARKER_HIP_ID, self.MARKER_KNEE_ID)
        knee_angle = knee_angle_deg(centers, self.MARKER_HIP_ID, self.MARKER_KNEE_ID, self.MARKER_ANKLE_ID)

        return femur_angle, knee_angle, depth_angle, self.bar_height_px

    # ------------------------------------------------------------------
    # Depth Stability / Hysterese + Mindestdauer
    # ------------------------------------------------------------------
    def update_depth_stability(self, depth_angle):
        """
        Liefert:
        - depth_class: "OK" / "HIGH" / "MID" / None
        - ok_stable: True wenn OK mindestens stable_frames_required Frames
        - high_stable: True wenn HIGH mindestens stable_frames_required Frames
        """
        if depth_angle is None:
            self._ok_streak = 0
            self._high_streak = 0
            return None, False, False

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
            self._ok_streak = 0
            self._high_streak = 0

        ok_stable = self._ok_streak >= self.stable_frames_required
        high_stable = self._high_streak >= self.stable_frames_required
        return depth_class, ok_stable, high_stable
    
    # ------------------------------------------------------------------
    # Sound bei gültigem Squat
    # ------------------------------------------------------------------
    def play_valid_squat_sound(self):
        """
        Spielt einen kurzen Sound ab (wenn verfügbar).
        """
        if not self.sound_enabled_var.get():
            return

        # Windows-Beep (am zuverlässigsten ohne externe libs)
        if winsound is not None:
            # frequency Hz, duration ms
            winsound.Beep(880, 150)
            return

        # Fallback: Terminal bell (kann funktionieren, je nach System/Terminal)
        sys.stdout.write("\a")
        sys.stdout.flush()


    # ------------------------------------------------------------------
    # Tracking Status
    # ------------------------------------------------------------------
    def update_tracking_status(self, centers: dict):
        required = {
            self.MARKER_HIP_ID: "HIP",
            self.MARKER_KNEE_ID: "KNEE",
            self.MARKER_ANKLE_ID: "ANKLE",
            self.MARKER_BAR_ID: "BAR",
        }

        present = [mid for mid in required.keys() if mid in centers]
        missing = [required[mid] for mid in required.keys() if mid not in centers]

        if len(present) == 3:
            status = "OK"
        elif len(present) == 2:
            status = "DEGRADED"
        elif len(present) == 1:
            status = "WEAK"
        else:
            status = "LOST"

        if missing:
            self.tracking_status_var.set(f"Tracking: {status}  (missing: {', '.join(missing)})")
        else:
            self.tracking_status_var.set("Tracking: OK  (3/3 markers)")

    # ------------------------------------------------------------------
    # Visualisierung (Linien/Punkte)
    # ------------------------------------------------------------------
    def draw_segments(self, frame):
        centers = self.last_centers

        hip_pt = knee_pt = ankle_pt = None

        has_hip = self.MARKER_HIP_ID in centers
        has_knee = self.MARKER_KNEE_ID in centers
        has_ankle = self.MARKER_ANKLE_ID in centers
        has_bar = self.MARKER_BAR_ID in centers


        if has_hip:
            hip = centers[self.MARKER_HIP_ID]
            hip_pt = (int(hip[0]), int(hip[1]))
            cv2.circle(frame, hip_pt, 6, (0, 0, 255), -1)  # rot

        if has_knee:
            knee = centers[self.MARKER_KNEE_ID]
            knee_pt = (int(knee[0]), int(knee[1]))
            cv2.circle(frame, knee_pt, 6, (255, 0, 0), -1)  # blau

        if has_ankle:
            ankle = centers[self.MARKER_ANKLE_ID]
            ankle_pt = (int(ankle[0]), int(ankle[1]))
            cv2.circle(frame, ankle_pt, 6, (0, 255, 0), -1)  # grün


        # Oberschenkel rot
        if has_hip and has_knee and hip_pt and knee_pt:
            cv2.line(frame, hip_pt, knee_pt, (0, 0, 255), 2)

        # Unterschenkel grün
        if has_knee and has_ankle and knee_pt and ankle_pt:
            cv2.line(frame, knee_pt, ankle_pt, (0, 255, 0), 2)

        # Bar/Handle Marker (ID 43) = gelb + horizontale Linie
        if has_bar:
            bar = centers[self.MARKER_BAR_ID]
            bar_pt = (int(bar[0]), int(bar[1]))

            # Punkt
            cv2.circle(frame, bar_pt, 7, (0, 255, 255), -1)  # gelb (BGR)

            # horizontale Linie auf Bar-Höhe
            h, w, _ = frame.shape
            cv2.line(frame, (0, bar_pt[1]), (w, bar_pt[1]), (0, 255, 255), 2)

            # Label
            cv2.putText(frame, "BAR (43)", (bar_pt[0] + 10, bar_pt[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)


if __name__ == "__main__":
    app = SquatAnalysisApp(camera_index=1)
    app.run()