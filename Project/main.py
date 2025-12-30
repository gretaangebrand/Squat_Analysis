import sys
import cv2
import tkinter as tk
from tkinter import ttk
from collections import deque
import winsound
import numpy as np

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from aruco_tracker import ArucoTracker, ArucoTrackerConfig
from angles import femur_segment_angle_deg, squat_depth_angle_deg, knee_angle_deg, knee_valid_angle_deg


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
    - Holt Markerzentren über ArucoTracker (ausgelagert, inkl. max_gap_seconds)
    - Berechnet Winkel + Bar-Höhe
    - Zeichnet Segmente + zeigt Werte in GUI
    - Tracking Status + stabiler Valid-Squat Counter (State Machine)
    - Live Plots (Knee + Bar)
    - Histogram Tab (wird beim Stop befüllt, Standpose herausgefiltert)
    """

    # Marker IDs (anpassen je nach Platzierung)
    MARKER_HIP_ID = 42
    MARKER_KNEE_ID = 41
    MARKER_ANKLE_ID = 40
    MARKER_BAR_ID = 43  # Handle/Bar marker

    def __init__(self, camera_index: int = 1):
        # -------------------------
        # GUI Setup
        # -------------------------
        self.window = tk.Tk()
        self.window.title("Squat Analysis")

        # -------------------------
        # Tabs (Notebook)
        # -------------------------
        self.notebook = ttk.Notebook(self.window)
        self.tab_live = ttk.Frame(self.notebook)
        self.tab_hist = ttk.Frame(self.notebook)

        self.notebook.add(self.tab_live, text="Live")
        self.notebook.add(self.tab_hist, text="Histogram")
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # -------------------------
        # Tk Variables / Labels (Live Tab)
        # -------------------------
        self.femur_angle_var = tk.StringVar(value="Femur angle: -- °")
        self.knee_valid_angle_var = tk.StringVar(value="Knee angle: -- °")
        self.squat_status_var = tk.StringVar(value="Squat: --")
        self.tracking_status_var = tk.StringVar(value="Tracking: --")

        # Handle / Bar height tracking
        self.bar_y_ref = None
        self.bar_height_px = None
        self.bar_height_var = tk.StringVar(value="Bar height: -- px")

        ttk.Label(self.tab_live, textvariable=self.bar_height_var, font=("Arial", 16)).pack(pady=10)
        ttk.Label(self.tab_live, textvariable=self.femur_angle_var, font=("Arial", 16)).pack(pady=10)
        ttk.Label(self.tab_live, textvariable=self.knee_valid_angle_var, font=("Arial", 16)).pack(pady=10)
        ttk.Label(self.tab_live, textvariable=self.squat_status_var, font=("Arial", 16)).pack(pady=10)
        ttk.Label(self.tab_live, textvariable=self.tracking_status_var, font=("Arial", 14)).pack(pady=6)

        self.is_measuring = False
        ttk.Button(self.tab_live, text="Start measurement", command=self.start_measurement).pack(pady=10)
        ttk.Button(self.tab_live, text="Stop measurement", command=self.stop_measurement).pack(pady=5)

        # Knee-valid thresholds (signierter Winkel)
        # Idee: <= 0 bedeutet "mindestens 90° Beugung" erreicht (wie du es willst)
        self.knee_ok_threshold = 0.0
        self.knee_high_threshold = 10.0   # wenn Knie deutlich "zu offen" -> HIGH

        # Streaks für Knee zusätzlich (wie bei depth)
        self._knee_ok_streak = 0
        self._knee_high_streak = 0

        # State + Flag, ob unten wirklich gültig war
        self._squat_state = "TOP"          # TOP / BOTTOM
        self._bottom_was_valid = False

        # optional: Cooldown nach Count, verhindert Doppelt-Zählen bei wackeliger TOP-Phase
        self._cooldown_frames = 0
        self.cooldown_after_rep = 10       # ~0.4s bei 25fps


        # -------------------------
        # Squat Counter / State Machine (Live Tab)
        # -------------------------
        self.rep_count = 0
        self.rep_count_var = tk.StringVar(value="Valid squats: 0")
        ttk.Label(self.tab_live, textvariable=self.rep_count_var, font=("Arial", 16)).pack(pady=10)

        # Sound checkbox
        self.sound_enabled_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            self.tab_live,
            text="Sound on valid squat",
            variable=self.sound_enabled_var
        ).pack(pady=6)

        # Hysterese
        self.depth_ok_threshold = -2.0
        self.depth_high_threshold = 2.0

        # Mindestdauer (Frames)
        self.stable_frames_required = 5
        self._ok_streak = 0
        self._high_streak = 0
        self._squat_state = "TOP"

        # -------------------------
        # Kamera Setup
        # -------------------------
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            print("❌ Cannot open camera")
        else:
            print("Camera opened successfully")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        self.update_interval = 40  # ms ≈ 25 fps GUI loop
        print(f"GUI update interval (ms): {self.update_interval}")

        # -------------------------
        # ArUco Tracker (ausgelagert)
        # -------------------------
        tracker_cfg = ArucoTrackerConfig(
            dictionary=cv2.aruco.DICT_6X6_250,
            update_interval_ms=self.update_interval,
            max_gap_seconds=1.0,
        )
        self.tracker = ArucoTracker(tracker_cfg)

        self.last_centers = {}

        # -------------------------
        # Live Plot Buffers
        # -------------------------
        self.knee_angle_buffer = deque(maxlen=MAX_SAMPLES)
        self.bar_height_buffer = deque(maxlen=MAX_SAMPLES)

        # -------------------------
        # ---- Live Figure ----
        self.fig = Figure(figsize=(7, 4.5), dpi=100)

        self.ax_knee = self.fig.add_subplot(211)
        self.knee_line, = self.ax_knee.plot([], [], linewidth=2)

        self.ax_bar = self.fig.add_subplot(212)
        self.bar_line, = self.ax_bar.plot([], [], linewidth=2)

        # Axis styling AFTER axes exist
        self.ax_knee.set_title(f"Knee Angle (last {WINDOW_SECONDS} seconds)")
        self.ax_knee.set_xlabel("Time (s)")
        self.ax_knee.set_ylabel("Angle (deg)")
        self.ax_knee.set_ylim(-60, 90)
        self.ax_knee.set_xlim(-WINDOW_SECONDS, 0)
        self.ax_knee.grid(True)

        self.ax_bar.set_title(f"Bar Height (last {WINDOW_SECONDS} seconds)")
        self.ax_bar.set_xlabel("Time (s)")
        self.ax_bar.set_ylabel("Depth (px)")
        self.ax_bar.set_xlim(-WINDOW_SECONDS, 0)
        self.ax_bar.set_ylim(0, 300)
        self.ax_bar.grid(True)

        self.fig.tight_layout()

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.tab_live)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, pady=10)


        # -------------------------
        # Histogram Figure (Tab) (wird beim Stop befüllt)
        # -------------------------
        # Histogram plot will be created on Stop
        self.hist_fig = None
        self.hist_canvas = None
        self.ax_hist_depth = None
        self.ax_hist_xy = None


        # -------------------------
        # Data for histogram (collected during session)
        # -------------------------
        self.hist_depth_samples = []
        self.hist_x_samples = []
        self.hist_y_samples = []
        self.stand_depth_threshold_px = 10.0  # ggf. tunen

        # sauberes Beenden
        self.window.protocol("WM_DELETE_WINDOW", self.on_close)

    # ------------------------------------------------------------------
    # GUI Steuerung
    # ------------------------------------------------------------------
    def start_measurement(self):
        if not self.is_measuring:
            self.tracker.reset()

            # counter/state reset
            self.rep_count = 0
            self.rep_count_var.set("Valid squats: 0")
            self._squat_state = "TOP"
            self._ok_streak = 0
            self._high_streak = 0

            # bar reference reset
            self.bar_y_ref = None
            self.bar_height_px = None
            self.bar_height_var.set("Bar height: -- px")

            # buffers reset
            self.knee_angle_buffer.clear()
            self.bar_height_buffer.clear()

            # histogram data reset
            self.hist_depth_samples.clear()
            self.hist_x_samples.clear()
            self.hist_y_samples.clear()

            self.is_measuring = True
            self.update_loop()
            print("Measurement started")

    def stop_measurement(self):
        self.is_measuring = False
        print("Measurement stopped")

        # render histograms and switch tab
        self.render_histograms()
        self.notebook.select(self.tab_hist)


    def on_close(self):
        self.is_measuring = False
        if self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()
        self.window.destroy()

    def run(self):
        self.window.mainloop()

    # ------------------------------------------------------------------
    # Main Loop
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

            femur_angle, knee_valid, squat_depth_angle, bar_height_px = self.process_frame(frame)


            # Tracking Status
            self.update_tracking_status(self.last_centers)

            # GUI Winkel
            self.femur_angle_var.set(f"Femur angle: {femur_angle:.1f} °" if femur_angle is not None else "Femur angle: -- °")
            self.knee_valid_angle_var.set(f"Knee angle: {knee_angle:.1f} °" if knee_angle is not None else "Knee angle: -- °")
            self.bar_height_var.set(f"Bar height: {bar_height_px:.1f} px" if bar_height_px is not None else "Bar height: -- px")

            # --- Stabilisierung + Statusanzeige + Counter ---
            depth_class, ok_stable, high_stable = self.update_depth_stability(squat_depth_angle)

            # Cooldown runterzählen
            if self._cooldown_frames > 0:
                self._cooldown_frames -= 1

            # Kombinierte Stabilisierung (Depth + Knee_valid)
            cls, ok_stable, high_stable = self.update_validity_stability(squat_depth_angle, knee_valid)

            # Anzeige
            if cls is None:
                self.squat_status_var.set("Squat: --")
            elif cls == "OK":
                self.squat_status_var.set("Squat: ✅ depth OK")
            elif cls == "HIGH":
                self.squat_status_var.set("Squat: ❌ too high")
            else:
                self.squat_status_var.set("Squat: ⚠ borderline")

            # Wenn Tracking weg -> state reset (wichtig!)
            if cls is None:
                self._squat_state = "TOP"
                self._bottom_was_valid = False

            # -------------------------
            # State Machine: TOP <-> BOTTOM
            # -------------------------
            if self._cooldown_frames == 0:
                # TOP -> BOTTOM wenn unten stabil erreicht
                if self._squat_state == "TOP":
                    if ok_stable:
                        self._squat_state = "BOTTOM"
                        self._bottom_was_valid = True  # wir waren unten gültig

                # BOTTOM -> TOP wenn oben stabil erreicht
                elif self._squat_state == "BOTTOM":
                    if high_stable:
                        # Zählen nur wenn unten gültig
                        if self._bottom_was_valid:
                            self.rep_count += 1
                            self.rep_count_var.set(f"Valid squats: {self.rep_count}")
                            self.play_valid_squat_sound()

                        # Reset
                        self._squat_state = "TOP"
                        self._bottom_was_valid = False
                        self._cooldown_frames = self.cooldown_after_rep


            # --- Buffers updaten ---
            self.knee_angle_buffer.append(knee_valid if knee_valid is not None else float("nan"))
            self.bar_height_buffer.append(bar_height_px if bar_height_px is not None else float("nan"))


            # --- Live Plots updaten ---
            self.update_live_plots()

            # --- Collect histogram data (exclude stand pose) ---
            if (self.MARKER_BAR_ID in self.last_centers) and (bar_height_px is not None):
                bar = self.last_centers[self.MARKER_BAR_ID]
                bar_x = float(bar[0])
                bar_y = float(bar[1])

                # Standpose raus: nur Werte, die "wirklich tiefer" sind
                if bar_height_px > self.stand_depth_threshold_px:
                    self.hist_depth_samples.append(float(bar_height_px))
                    self.hist_x_samples.append(bar_x)
                    self.hist_y_samples.append(bar_y)

            # Boden-Referenzlinie (optional)
            h, w, _ = frame.shape
            y_floor = int(h * 0.8)
            cv2.line(frame, (0, y_floor), (w, y_floor), (0, 255, 0), 2)

            # Segmente + Bar Marker zeichnen
            self.draw_segments(frame)

            # Overlay
            cv2.putText(frame, f"Reps: {self.rep_count}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            cv2.putText(frame, f"State: {self._squat_state}", (10, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            if bar_height_px is not None:
                cv2.putText(frame, f"Bar depth: {bar_height_px:.1f}px", (10, 95),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            # Frame anzeigen
            cv2.imshow("Squat Camera", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                self.stop_measurement()

            self.window.after(self.update_interval, self.update_loop)

        except Exception as e:
            print(f"❌ Error in update_loop: {e}")
            self.squat_status_var.set("Squat: ERROR (see console)")
            self.stop_measurement()

    # ------------------------------------------------------------------
    # Process frame: marker -> angles + bar height
    # ------------------------------------------------------------------
    def process_frame(self, frame):
        centers = self.tracker.update(frame, draw=True)
        self.last_centers = centers

        femur_angle = femur_segment_angle_deg(centers, self.MARKER_HIP_ID, self.MARKER_KNEE_ID)
        depth_angle = squat_depth_angle_deg(centers, self.MARKER_HIP_ID, self.MARKER_KNEE_ID)
        knee_valid = knee_valid_angle_deg(centers, self.MARKER_HIP_ID, self.MARKER_KNEE_ID, self.MARKER_ANKLE_ID, reference_deg=90.0)

        # --- Bar depth (positive = going down) ---
        bar_height_px = None
        if self.MARKER_BAR_ID in centers:
            bar_y = float(centers[self.MARKER_BAR_ID][1])
            if self.bar_y_ref is None:
                self.bar_y_ref = bar_y
            bar_height_px = bar_y - self.bar_y_ref  # >0 means lower than start

        self.bar_height_px = bar_height_px
        return femur_angle, knee_valid, depth_angle, bar_height_px


    # ------------------------------------------------------------------
    # Live plot update
    # ------------------------------------------------------------------
    def update_live_plots(self):
        # Convert buffers to lists
        knee_y = list(self.knee_angle_buffer)
        bar_y = list(self.bar_height_buffer)

        n = max(len(knee_y), len(bar_y))
        if n == 0:
            return

        # Time axis: from -WINDOW_SECONDS to 0 (last seconds)
        # We map samples evenly into the window (good enough even if FPS jitters a bit)
        t = np.linspace(-WINDOW_SECONDS, 0, n)

        # If one buffer is shorter (early in the run), pad with NaN so lengths match
        if len(knee_y) < n:
            knee_y = [float("nan")] * (n - len(knee_y)) + knee_y
        if len(bar_y) < n:
            bar_y = [float("nan")] * (n - len(bar_y)) + bar_y

        # Update line data
        self.knee_line.set_data(t, knee_y)
        self.bar_line.set_data(t, bar_y)

        # Fixed x-limits in seconds
        self.ax_knee.set_xlim(-WINDOW_SECONDS, 0)
        self.ax_bar.set_xlim(-WINDOW_SECONDS, 0)

        # Knee y-limits (keep fixed, readable)
        self.ax_knee.set_ylim(-60, 90)

        # Bar y-limits: dynamic but stable (ignore NaNs)
        finite_bar = [v for v in bar_y if v == v]  # NaN != NaN
        if finite_bar:
            vmin = min(finite_bar)
            vmax = max(finite_bar)
            # ensure we always show at least a sensible range
            pad = max(10.0, 0.15 * (vmax - vmin + 1e-9))
            low = max(0.0, vmin - pad)             # depth should not go far below 0
            high = vmax + pad
            # if you're barely moving, avoid micro-scale like 0.05 px
            if high - low < 50:
                high = low + 50
            self.ax_bar.set_ylim(low, high)
        else:
            self.ax_bar.set_ylim(0, 300)

        self.canvas.draw_idle()


    # ------------------------------------------------------------------
    # Depth stability / hysterese + minimum duration
    # ------------------------------------------------------------------
    def update_depth_stability(self, depth_angle):
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
    
    def update_validity_stability(self, depth_angle, knee_valid):
        """
        Kombiniert Depth + Knee_valid zu einer stabilen Klassifikation.
        Liefert:
        - cls: None / "OK" / "HIGH" / "MID"
        - ok_stable: True/False
        - high_stable: True/False
        """

        # Wenn Depth fehlt -> keine sichere Klassifikation (Tracking/Marker)
        if depth_angle is None:
            self._ok_streak = 0
            self._high_streak = 0
            self._knee_ok_streak = 0
            self._knee_high_streak = 0
            return None, False, False

        # Depth-Klasse (mit Hysterese)
        if depth_angle <= self.depth_ok_threshold:
            depth_cls = "OK"
        elif depth_angle >= self.depth_high_threshold:
            depth_cls = "HIGH"
        else:
            depth_cls = "MID"

        # Knee-Klasse (falls verfügbar)
        if knee_valid is None:
            knee_cls = "MID"   # konservativ: ohne Knieinfo nicht "OK" zählen
        else:
            if knee_valid <= self.knee_ok_threshold:
                knee_cls = "OK"
            elif knee_valid >= self.knee_high_threshold:
                knee_cls = "HIGH"
            else:
                knee_cls = "MID"

        # Kombi-Entscheidung (konservativ)
        # OK nur wenn beide OK
        if depth_cls == "OK" and knee_cls == "OK":
            cls = "OK"
        # HIGH wenn entweder klar HIGH (zu hoch)
        elif depth_cls == "HIGH" or knee_cls == "HIGH":
            cls = "HIGH"
        else:
            cls = "MID"

        # Streak-Update (stabilisieren)
        if cls == "OK":
            self._ok_streak += 1
            self._high_streak = 0
        elif cls == "HIGH":
            self._high_streak += 1
            self._ok_streak = 0
        else:
            self._ok_streak = 0
            self._high_streak = 0

        ok_stable = self._ok_streak >= self.stable_frames_required
        high_stable = self._high_streak >= self.stable_frames_required
        return cls, ok_stable, high_stable


    # ------------------------------------------------------------------
    # Tracking status
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

        if len(present) == 4:
            status = "OK"
        elif len(present) == 3:
            status = "DEGRADED"
        elif len(present) == 2:
            status = "WEAK"
        else:
            status = "LOST"

        if missing:
            self.tracking_status_var.set(f"Tracking: {status}  (missing: {', '.join(missing)})")
        else:
            self.tracking_status_var.set("Tracking: OK  (4/4 markers)")

    # ------------------------------------------------------------------
    # Visualization: segments + bar marker
    # ------------------------------------------------------------------
    def draw_segments(self, frame):
        centers = self.last_centers

        hip_pt = knee_pt = ankle_pt = bar_pt = None

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

        # Femur
        if has_hip and has_knee and hip_pt and knee_pt:
            cv2.line(frame, hip_pt, knee_pt, (0, 0, 255), 2)

        # Tibia
        if has_knee and has_ankle and knee_pt and ankle_pt:
            cv2.line(frame, knee_pt, ankle_pt, (0, 255, 0), 2)

        # Bar marker + horizontal line
        if has_bar:
            bar = centers[self.MARKER_BAR_ID]
            bar_pt = (int(bar[0]), int(bar[1]))
            cv2.circle(frame, bar_pt, 7, (0, 255, 255), -1)  # gelb

            h, w, _ = frame.shape
            cv2.line(frame, (0, bar_pt[1]), (w, bar_pt[1]), (0, 255, 255), 2)
            cv2.putText(frame, "BAR (43)", (bar_pt[0] + 10, bar_pt[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # ------------------------------------------------------------------
    # Histogram rendering (called on stop)
    # ------------------------------------------------------------------
    def render_histograms(self):
        # --- old histogram canvas entfernen (falls vorhanden) ---
        if self.hist_canvas is not None:
            self.hist_canvas.get_tk_widget().destroy()
            self.hist_canvas = None
            self.hist_fig = None
            self.ax_hist_depth = None
            self.ax_hist_xy = None

        # --- neue Figure erstellen ---
        self.hist_fig = Figure(figsize=(7, 4.5), dpi=100)
        self.ax_hist_depth = self.hist_fig.add_subplot(121)
        self.ax_hist_xy = self.hist_fig.add_subplot(122)

        # Titles/Labels
        self.ax_hist_depth.set_title("Bar depth distribution (filtered)")
        self.ax_hist_depth.set_xlabel("Depth (px)")
        self.ax_hist_depth.set_ylabel("Count")
        self.ax_hist_depth.grid(True)

        self.ax_hist_xy.set_title("Bar position distribution (x,y)")
        self.ax_hist_xy.set_xlabel("x (px)")
        self.ax_hist_xy.set_ylabel("y (px)")
        self.ax_hist_xy.grid(True)

        # --- Check Data ---
        if len(self.hist_depth_samples) < 5:
            self.ax_hist_depth.text(
                0.5, 0.5,
                "Not enough data collected\n(check threshold / marker visibility)",
                ha="center", va="center",
                transform=self.ax_hist_depth.transAxes
            )
            self.ax_hist_xy.text(
                0.5, 0.5,
                "Not enough data collected",
                ha="center", va="center",
                transform=self.ax_hist_xy.transAxes
            )
        else:
            # 1D Histogram: depth
            self.ax_hist_depth.hist(self.hist_depth_samples, bins=20)

            # 2D Histogram: x/y position
            self.ax_hist_xy.hist2d(self.hist_x_samples, self.hist_y_samples, bins=30)

        self.hist_fig.tight_layout()

        # --- Canvas in Histogram-Tab einbetten ---
        self.hist_canvas = FigureCanvasTkAgg(self.hist_fig, master=self.tab_hist)
        self.hist_canvas.draw()
        self.hist_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, pady=10)


    # ------------------------------------------------------------------
    # Sound
    # ------------------------------------------------------------------
    def play_valid_squat_sound(self):
        if not self.sound_enabled_var.get():
            return

        if winsound is not None:
            winsound.Beep(880, 150)
            return

        # fallback: terminal bell
        sys.stdout.write("\a")
        sys.stdout.flush()


if __name__ == "__main__":
    app = SquatAnalysisApp(camera_index=1)
    app.run()
