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
from angles import (
    femur_segment_angle_deg,
    squat_depth_angle_deg,
    knee_valid_angle_deg,
)

# =========================
# Plot configuration
# =========================
WINDOW_SECONDS = 6
FPS_ESTIMATE = 25
MAX_SAMPLES = WINDOW_SECONDS * FPS_ESTIMATE  # 150


class SquatAnalysisApp:
    """
    Squat Analysis – Main GUI App

    - Kamera + ArUco Tracking (inkl. max_gap_seconds)
    - Winkel + Bar-Höhe
    - Stabilisierung + State Machine Counter
    - Live Plots (Knee + Bar height)
    - Histogram Tab: Ring-Plot + X/Y-Hist (nur Bewegung, Standphase raus)
    """

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
        # Tabs
        # -------------------------
        self.notebook = ttk.Notebook(self.window)
        self.tab_live = ttk.Frame(self.notebook)
        self.tab_hist = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_live, text="Live")
        self.notebook.add(self.tab_hist, text="Histogram")
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # -------------------------
        # Tk Variables (Live)
        # -------------------------
        self.femur_angle_var = tk.StringVar(value="Femur angle: -- °")
        self.knee_valid_angle_var = tk.StringVar(value="Knee angle: -- °")
        self.squat_status_var = tk.StringVar(value="Squat: --")
        self.tracking_status_var = tk.StringVar(value="Tracking: --")

        self.bar_y_ref = None
        self.bar_height_var = tk.StringVar(value="Bar height: --")

        ttk.Label(self.tab_live, textvariable=self.bar_height_var, font=("Arial", 16)).pack(pady=10)
        ttk.Label(self.tab_live, textvariable=self.femur_angle_var, font=("Arial", 16)).pack(pady=10)
        ttk.Label(self.tab_live, textvariable=self.knee_valid_angle_var, font=("Arial", 16)).pack(pady=10)
        ttk.Label(self.tab_live, textvariable=self.squat_status_var, font=("Arial", 16)).pack(pady=10)
        ttk.Label(self.tab_live, textvariable=self.tracking_status_var, font=("Arial", 14)).pack(pady=6)

        self.is_measuring = False
        ttk.Button(self.tab_live, text="Start measurement", command=self.start_measurement).pack(pady=10)
        ttk.Button(self.tab_live, text="Stop measurement", command=self.stop_measurement).pack(pady=5)

        # -------------------------
        # Counter + Sound
        # -------------------------
        self.rep_count = 0
        self.rep_count_var = tk.StringVar(value="Valid squats: 0")
        ttk.Label(self.tab_live, textvariable=self.rep_count_var, font=("Arial", 16)).pack(pady=10)

        self.sound_enabled_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(self.tab_live, text="Sound on valid squat", variable=self.sound_enabled_var).pack(pady=6)

        # -------------------------
        # Validity thresholds (combined depth + knee_valid)
        # -------------------------
        # Valid squat: depth <= 0 AND knee_valid <= 0 (your definition)
        self.stable_frames_required = 5
        self._ok_streak = 0
        self._high_streak = 0

        self._squat_state = "TOP"          # TOP / BOTTOM
        self._bottom_was_valid = False

        self._cooldown_frames = 0
        self.cooldown_after_rep = 10  # ~0.4s @25fps

        # -------------------------
        # Camera
        # -------------------------
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            print("❌ Cannot open camera")
        else:
            print("Camera opened successfully")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        self.update_interval = 40  # ms ~ 25fps
        print(f"GUI update interval (ms): {self.update_interval}")

        # -------------------------
        # ArUco tracker
        # -------------------------
        tracker_cfg = ArucoTrackerConfig(
            dictionary=cv2.aruco.DICT_6X6_250,
            update_interval_ms=self.update_interval,
            max_gap_seconds=1.0,
        )
        self.tracker = ArucoTracker(tracker_cfg)
        self.last_centers = {}

        # -------------------------
        # Live buffers (units depend on calibration availability)
        # -------------------------
        self.knee_angle_buffer = deque(maxlen=MAX_SAMPLES)
        self.bar_height_buffer = deque(maxlen=MAX_SAMPLES)

        # -------------------------
        # Live figure (in LIVE tab)
        # -------------------------
        self.fig = Figure(figsize=(7, 4.5), dpi=100)

        self.ax_knee = self.fig.add_subplot(211)
        self.knee_line, = self.ax_knee.plot([], [], linewidth=2)

        self.ax_bar = self.fig.add_subplot(212)
        self.bar_line, = self.ax_bar.plot([], [], linewidth=2)

        self.ax_knee.set_title(f"Knee Angle (last {WINDOW_SECONDS} seconds)")
        self.ax_knee.set_xlabel("Time (s)")
        self.ax_knee.set_ylabel("Angle (deg)")
        self.ax_knee.set_ylim(-60, 90)
        self.ax_knee.set_xlim(-WINDOW_SECONDS, 0)
        self.ax_knee.grid(True)

        self.ax_bar.set_title(f"Bar Height (last {WINDOW_SECONDS} seconds)")
        self.ax_bar.set_xlabel("Time (s)")
        self.ax_bar.set_ylabel("Height")  # label updated dynamically (px vs cm)
        self.ax_bar.set_xlim(-WINDOW_SECONDS, 0)
        self.ax_bar.grid(True)

        self.fig.tight_layout()

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.tab_live)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, pady=10)

        # -------------------------
        # Histogram figure (created on Stop)
        # -------------------------
        self.hist_fig = None
        self.hist_canvas = None

        # -------------------------
        # Ring-plot data (movement only)
        # -------------------------
        self.bar_path_x = []
        self.bar_path_y = []

        # stand/motion thresholds (in SAME UNIT as bar_height we store)
        # -> we apply these in PX domain before conversion
        self.stand_depth_threshold_px = 10.0
        self.motion_threshold_px = 1.5
        self._prev_bar_height_px = None

        # Clean exit
        self.window.protocol("WM_DELETE_WINDOW", self.on_close)

    # -------------------------
    # GUI control
    # -------------------------
    def start_measurement(self):
        if self.is_measuring:
            return

        self.tracker.reset()

        # reset counter/state
        self.rep_count = 0
        self.rep_count_var.set("Valid squats: 0")
        self._squat_state = "TOP"
        self._bottom_was_valid = False
        self._ok_streak = 0
        self._high_streak = 0
        self._cooldown_frames = 0

        # bar ref
        self.bar_y_ref = None
        self.bar_height_var.set("Bar height: --")

        # buffers
        self.knee_angle_buffer.clear()
        self.bar_height_buffer.clear()

        # ring data
        self.bar_path_x.clear()
        self.bar_path_y.clear()
        self._prev_bar_height_px = None

        self.is_measuring = True
        self.update_loop()
        print("Measurement started")

    def stop_measurement(self):
        self.is_measuring = False
        print("Measurement stopped")

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

    # -------------------------
    # Main loop
    # -------------------------
    def update_loop(self):
        if not self.is_measuring:
            return

        try:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to grab frame")
                self.window.after(100, self.update_loop)
                return

            femur_angle, knee_valid, depth_angle, bar_height_px = self.process_frame(frame)

            # tracking status
            self.update_tracking_status(self.last_centers)

            # UI
            self.femur_angle_var.set(f"Femur angle: {femur_angle:.1f} °" if femur_angle is not None else "Femur angle: -- °")
            self.knee_valid_angle_var.set(f"Knee angle: {knee_valid:.1f} °" if knee_valid is not None else "Knee angle: -- °")

            # bar height display + choose unit for live plot
            bar_height_value, bar_unit = self.bar_height_to_display(bar_height_px)
            if bar_height_value is None:
                self.bar_height_var.set("Bar height: --")
            else:
                self.bar_height_var.set(f"Bar height: {bar_height_value:.1f} {bar_unit}")

            # validity stability (combined)
            cls, ok_stable, high_stable = self.update_validity_stability(depth_angle, knee_valid)

            # cooldown
            if self._cooldown_frames > 0:
                self._cooldown_frames -= 1

            # status label
            if cls is None:
                self.squat_status_var.set("Squat: --")
            elif cls == "OK":
                self.squat_status_var.set("Squat: ✅ valid")
            elif cls == "HIGH":
                self.squat_status_var.set("Squat: ❌ invalid")
            else:
                self.squat_status_var.set("Squat: ⚠ borderline")

            # tracking lost -> reset state
            if cls is None:
                self._squat_state = "TOP"
                self._bottom_was_valid = False

            # state machine (count once per rep)
            if self._cooldown_frames == 0:
                if self._squat_state == "TOP":
                    if ok_stable:
                        self._squat_state = "BOTTOM"
                        self._bottom_was_valid = True
                elif self._squat_state == "BOTTOM":
                    if high_stable:
                        if self._bottom_was_valid:
                            self.rep_count += 1
                            self.rep_count_var.set(f"Valid squats: {self.rep_count}")
                            self.play_valid_squat_sound()
                        self._squat_state = "TOP"
                        self._bottom_was_valid = False
                        self._cooldown_frames = self.cooldown_after_rep

            # buffers update (knee always in deg, bar in chosen unit)
            self.knee_angle_buffer.append(knee_valid if knee_valid is not None else float("nan"))
            self.bar_height_buffer.append(bar_height_value if bar_height_value is not None else float("nan"))

            # live plots
            self.update_live_plots(bar_unit=bar_unit)

            # collect ring data (movement only, no stand)
            self.collect_ring_points(cls, bar_height_px)

            # draw overlays
            self.draw_live_feedback(frame, cls, tracking_text=self.tracking_status_var.get())
            self.draw_segments(frame)

            cv2.putText(frame, f"State: {self._squat_state}", (10, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            cv2.imshow("Squat Camera", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                self.stop_measurement()

            self.window.after(self.update_interval, self.update_loop)

        except Exception as e:
            print(f"❌ Error in update_loop: {e}")
            self.squat_status_var.set("Squat: ERROR (see console)")
            self.stop_measurement()

    # -------------------------
    # Process frame
    # -------------------------
    def process_frame(self, frame):
        centers = self.tracker.update(frame, draw=True)
        self.last_centers = centers

        femur_angle = femur_segment_angle_deg(centers, self.MARKER_HIP_ID, self.MARKER_KNEE_ID)
        depth_angle = squat_depth_angle_deg(centers, self.MARKER_HIP_ID, self.MARKER_KNEE_ID)
        knee_valid = knee_valid_angle_deg(
            centers,
            self.MARKER_HIP_ID, self.MARKER_KNEE_ID, self.MARKER_ANKLE_ID,
            reference_deg=90.0
        )

        bar_height_px = None
        if self.MARKER_BAR_ID in centers:
            bar_y = float(centers[self.MARKER_BAR_ID][1])
            if self.bar_y_ref is None:
                self.bar_y_ref = bar_y
            bar_height_px = bar_y - self.bar_y_ref  # >0 means lower than start

        return femur_angle, knee_valid, depth_angle, bar_height_px

    # -------------------------
    # Helpers: unit conversion
    # -------------------------
    def get_mm_per_px(self):
        # tracker may or may not provide mm_per_px; do safe access
        return getattr(self.tracker, "mm_per_px", None)

    def bar_height_to_display(self, bar_height_px):
        """
        returns (value, unit_str) where unit is either 'cm' (if calibrated) or 'px'.
        """
        if bar_height_px is None:
            return None, "px"

        mm_per_px = self.get_mm_per_px()
        if mm_per_px is not None:
            return (bar_height_px * mm_per_px) / 10.0, "cm"  # cm
        return float(bar_height_px), "px"

    # -------------------------
    # Live plots (seconds on x)
    # -------------------------
    def update_live_plots(self, bar_unit="px"):
        knee_y = list(self.knee_angle_buffer)
        bar_y = list(self.bar_height_buffer)

        n = max(len(knee_y), len(bar_y))
        if n == 0:
            return

        t = np.linspace(-WINDOW_SECONDS, 0, n)

        if len(knee_y) < n:
            knee_y = [float("nan")] * (n - len(knee_y)) + knee_y
        if len(bar_y) < n:
            bar_y = [float("nan")] * (n - len(bar_y)) + bar_y

        self.knee_line.set_data(t, knee_y)
        self.bar_line.set_data(t, bar_y)

        self.ax_knee.set_xlim(-WINDOW_SECONDS, 0)
        self.ax_bar.set_xlim(-WINDOW_SECONDS, 0)

        self.ax_knee.set_ylim(-60, 90)

        finite_bar = [v for v in bar_y if v == v]
        if finite_bar:
            vmin, vmax = min(finite_bar), max(finite_bar)
            pad = max(1.0 if bar_unit == "cm" else 10.0, 0.15 * (vmax - vmin + 1e-9))
            low = max(0.0, vmin - pad)
            high = vmax + pad
            if (high - low) < (5.0 if bar_unit == "cm" else 50.0):
                high = low + (5.0 if bar_unit == "cm" else 50.0)
            self.ax_bar.set_ylim(low, high)
        else:
            self.ax_bar.set_ylim(0, 30 if bar_unit == "cm" else 300)

        self.ax_bar.set_ylabel(f"Height ({bar_unit})")

        self.canvas.draw_idle()

    # -------------------------
    # Combined stability: depth + knee_valid
    # -------------------------
    def update_validity_stability(self, squat_depth_angle, knee_valid):
        if squat_depth_angle is None or knee_valid is None:
            self._ok_streak = 0
            self._high_streak = 0
            return None, False, False

        is_ok = (squat_depth_angle <= 0) and (knee_valid <= 0)

        if is_ok:
            self._ok_streak += 1
            self._high_streak = 0
        else:
            self._high_streak += 1
            self._ok_streak = 0

        ok_stable = self._ok_streak >= self.stable_frames_required
        high_stable = self._high_streak >= self.stable_frames_required

        if ok_stable:
            return "OK", True, False
        if high_stable:
            return "HIGH", False, True
        return "TRANSITION", False, False

    # -------------------------
    # Ring-plot movement collection (movement only)
    # -------------------------
    def collect_ring_points(self, cls, bar_height_px):
        if (self.MARKER_BAR_ID not in self.last_centers) or (bar_height_px is None):
            self._prev_bar_height_px = None
            return

        bar = self.last_centers[self.MARKER_BAR_ID]
        bar_x_px = float(bar[0])
        bar_y_px = float(bar[1])

        moving = False
        if self._prev_bar_height_px is not None:
            if abs(bar_height_px - self._prev_bar_height_px) >= self.motion_threshold_px:
                moving = True
        self._prev_bar_height_px = bar_height_px

        if (cls is not None) and (bar_height_px > self.stand_depth_threshold_px) and moving:
            mm_per_px = self.get_mm_per_px()
            if mm_per_px is not None:
                x = (bar_x_px * mm_per_px) / 10.0  # cm
                y = (bar_y_px * mm_per_px) / 10.0  # cm
            else:
                x, y = bar_x_px, bar_y_px
            self.bar_path_x.append(x)
            self.bar_path_y.append(y)

    # -------------------------
    # Tracking status
    # -------------------------
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
            self.tracking_status_var.set(f"Tracking: {status} (missing: {', '.join(missing)})")
        else:
            self.tracking_status_var.set("Tracking: OK (4/4 markers)")

    # -------------------------
    # Draw segments
    # -------------------------
    def draw_segments(self, frame):
        c = self.last_centers

        def pt(mid):
            v = c.get(mid, None)
            if v is None:
                return None
            return (int(v[0]), int(v[1]))

        hip_pt = pt(self.MARKER_HIP_ID)
        knee_pt = pt(self.MARKER_KNEE_ID)
        ankle_pt = pt(self.MARKER_ANKLE_ID)
        bar_pt = pt(self.MARKER_BAR_ID)

        if hip_pt:
            cv2.circle(frame, hip_pt, 6, (0, 0, 255), -1)
        if knee_pt:
            cv2.circle(frame, knee_pt, 6, (255, 0, 0), -1)
        if ankle_pt:
            cv2.circle(frame, ankle_pt, 6, (0, 255, 0), -1)

        if hip_pt and knee_pt:
            cv2.line(frame, hip_pt, knee_pt, (0, 0, 255), 2)
        if knee_pt and ankle_pt:
            cv2.line(frame, knee_pt, ankle_pt, (0, 255, 0), 2)

        if bar_pt:
            cv2.circle(frame, bar_pt, 7, (0, 255, 255), -1)
            h, w = frame.shape[:2]
            cv2.line(frame, (0, bar_pt[1]), (w, bar_pt[1]), (0, 255, 255), 2)
            cv2.putText(frame, "BAR (43)", (bar_pt[0] + 10, bar_pt[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # -------------------------
    # Live feedback overlay
    # -------------------------
    def draw_live_feedback(self, frame, cls, tracking_text=None):
        if cls == "OK":
            color = (0, 200, 0)
            text = "VALID (OK)"
        elif cls == "HIGH":
            color = (0, 0, 255)
            text = "INVALID"
        elif cls == "TRANSITION":
            color = (0, 215, 255)
            text = "BORDERLINE"
        else:
            color = (180, 0, 180)
            text = "TRACKING LOST"

        h, w = frame.shape[:2]
        cv2.rectangle(frame, (10, 10), (330, 70), color, -1)
        cv2.putText(frame, text, (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

        if tracking_text:
            cv2.putText(frame, tracking_text, (10, 95),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), color, 6)

    # -------------------------
    # Histogram: ring + marginal hist (on stop)
    # -------------------------
    def render_histograms(self):
        if self.hist_canvas is not None:
            self.hist_canvas.get_tk_widget().destroy()
            self.hist_canvas = None
            self.hist_fig = None

        self.hist_fig = Figure(figsize=(8.0, 5.0), dpi=100)

        gs = self.hist_fig.add_gridspec(
            2, 2,
            width_ratios=[4, 1],
            height_ratios=[1, 4],
            wspace=0.05,
            hspace=0.05
        )

        ax_histx = self.hist_fig.add_subplot(gs[0, 0])
        ax_main  = self.hist_fig.add_subplot(gs[1, 0], sharex=ax_histx)
        ax_histy = self.hist_fig.add_subplot(gs[1, 1], sharey=ax_main)

        if len(self.bar_path_x) < 10:
            ax_main.text(
                0.5, 0.5,
                "Not enough MOVEMENT data collected\n(check marker visibility / thresholds)",
                ha="center", va="center",
                transform=ax_main.transAxes
            )
            ax_main.set_axis_off()
        else:
            x = np.asarray(self.bar_path_x, dtype=float)
            y = np.asarray(self.bar_path_y, dtype=float)

            # Unit label
            unit = "cm" if self.get_mm_per_px() is not None else "px"

            ax_main.hist2d(x, y, bins=30)
            ax_main.plot(x, y, linewidth=1)

            ax_main.set_title("Bar path (movement only, no stand phase)")
            ax_main.set_xlabel(f"x ({unit})")
            ax_main.set_ylabel(f"y ({unit})")
            ax_main.grid(True)

            # image-y grows downward; if using px, invert for visual similarity
            if unit == "px":
                ax_main.invert_yaxis()

            ax_histx.hist(x, bins=30)
            ax_histx.set_ylabel("count")
            ax_histx.grid(True)
            for lbl in ax_histx.get_xticklabels():
                lbl.set_visible(False)

            ax_histy.hist(y, bins=30, orientation="horizontal")
            ax_histy.set_xlabel("count")
            ax_histy.grid(True)
            for lbl in ax_histy.get_yticklabels():
                lbl.set_visible(False)
            ax_histy.invert_xaxis()

        self.hist_fig.tight_layout()

        self.hist_canvas = FigureCanvasTkAgg(self.hist_fig, master=self.tab_hist)
        self.hist_canvas.draw()
        self.hist_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, pady=10)

    # -------------------------
    # Sound
    # -------------------------
    def play_valid_squat_sound(self):
        if not self.sound_enabled_var.get():
            return

        try:
            winsound.Beep(880, 150)
        except Exception:
            sys.stdout.write("\a")
            sys.stdout.flush()


if __name__ == "__main__":
    app = SquatAnalysisApp(camera_index=1)
    app.run()