import cv2
import tkinter as tk
from tkinter import ttk
from collections import deque
import numpy as np
import time  # für tic toc Profiling
from pygrabber.dshow_graph import FilterGraph
from PIL import Image, ImageTk
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from aruco_tracker import ArucoTracker, ArucoTrackerConfig
from angles import squat_depth_angle_deg, knee_valid_angle_deg, femur_angle_depth_signed_deg
from sound import play_valid_squat_sound as _play_valid_squat_sound


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

    # Marker IDs
    MARKER_HIP_ID = 42
    MARKER_KNEE_ID = 41
    MARKER_ANKLE_ID = 40 
    MARKER_BAR_ID = 38  # Handle/Bar marker ID
    MARKER_FLOOR_ID = 39 # Bodenmarker ID 


    def __init__(self):
        # -------------------------
        # GUI Setup
        # -------------------------
        self.window = tk.Tk()
        self.window.grid_rowconfigure(1, weight=1)
        self.window.grid_columnconfigure(0, weight=1)
        self.window.title("Squat Analysis")

        # -------------------------
        # Tabs (Notebook)
        # -------------------------
        self.notebook = ttk.Notebook(self.window)
        self.tab_live = ttk.Frame(self.notebook)
        self.tab_hist = ttk.Frame(self.notebook)

        self.notebook.add(self.tab_live, text="Live")
        self.notebook.add(self.tab_hist, text="Histogram")
        self.notebook.grid(row=1, column=0, sticky="nsew")

        # -------------------------
        # Tk Variables / Labels (Live Tab)
        # -------------------------
        self.femur_angle_var = tk.StringVar(value="Femur angle: -- °")
        self.knee_valid_angle_var = tk.StringVar(value="Knee angle: -- °")
        self.squat_status_var = tk.StringVar(value="Squat: --")
        self.tracking_status_var = tk.StringVar(value="Tracking: --")
        self.active_camera_var = tk.StringVar(value="Camera: --")
        self._floor_marker_seen = False

        # Handle / Bar height tracking
        self.bar_y_ref = None
        self._bar_ref_locked = False
        self._bar_ref_candidates = deque(maxlen=30)   # ~1–1.2 s bei 25 fps
        self._bar_still_range_px = 2.0    # wie ruhig ist "ruhig" (px)
        self._bar_ref_min_samples = 15    # min. Frames für Referenz
        self.bar_height_px = None
        self.bar_height_var = tk.StringVar(value="Bar height: -- cm")

        # -------------------------
        # Live Tab: 2-column layout (LEFT = variables, RIGHT = plots)
        # -------------------------
        self.tab_live.grid_rowconfigure(0, weight=1)
        self.tab_live.grid_columnconfigure(0, weight=0)  # left column: compact
        self.tab_live.grid_columnconfigure(1, weight=1)  # right column: expands

        self.left_panel = ttk.Frame(self.tab_live, padding=(10, 10))
        # --- LEFT TOP: control buttons ---
        self.left_controls = ttk.Frame(self.left_panel)
        self.left_controls.pack(anchor="nw", fill="x", pady=(0, 15))

        self.is_measuring = False
        ttk.Button(self.left_controls,text="Start measurement",command=self.start_measurement).pack(fill="x", pady=(0, 6))
        ttk.Button(self.left_controls,text="Stop measurement",command=self.stop_measurement).pack(fill="x")

        self.right_panel = ttk.Frame(self.tab_live, padding=(10, 10))

        self.left_panel.grid(row=0, column=0, sticky="nsw")
        self.right_panel.grid(row=0, column=1, sticky="nsew")

        self.right_panel.grid_rowconfigure(0, weight=1)
        self.right_panel.grid_columnconfigure(0, weight=1)

        # --- LEFT: variables + controls ---
        ttk.Label(self.left_panel, textvariable=self.femur_angle_var, font=("Arial", 16)).pack(anchor="w", pady=(0, 10))
        ttk.Label(self.left_panel, textvariable=self.knee_valid_angle_var, font=("Arial", 16)).pack(anchor="w", pady=(0, 10))
        ttk.Label(self.left_panel, textvariable=self.bar_height_var, font=("Arial", 16)).pack(anchor="w", pady=(0, 10))
        ttk.Label(self.left_panel, textvariable=self.squat_status_var, font=("Arial", 16)).pack(anchor="w", pady=(0, 10))
        ttk.Label(self.left_panel, textvariable=self.tracking_status_var, font=("Arial", 16)).pack(anchor="w", pady=(0, 10))
        ttk.Label(self.left_panel,textvariable=self.active_camera_var, font=("Arial", 16)).pack(anchor="w", pady=(0, 10))



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

        # Cooldown nach Count, verhindert Doppelt-Zählen bei wackeliger TOP-Phase
        self._cooldown_frames = 0
        self.cooldown_after_rep = 10       # ~0.4s bei 25fps

        # -------------------------
        # Squat Counter / State Machine (Live Tab)
        # -------------------------
        self.rep_count = 0
        self.rep_count_var = tk.StringVar(value="Valid squats: 0")
        ttk.Label(self.left_panel, textvariable=self.rep_count_var, font=("Arial", 16)).pack(anchor="w", pady=(0, 10))

        # Sound checkbox
        self.sound_enabled_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(self.left_panel,text="Sound on valid squat",variable=self.sound_enabled_var).pack(anchor="w", pady=(0, 6))

        # Hysterese
        self.depth_ok_threshold = -2.0
        self.depth_high_threshold = 2.0

        # Mindestdauer (Frames)
        self.stable_frames_required = 5
        self._ok_streak = 0
        self._high_streak = 0
        self._squat_state = "TOP"

        # --- Camera selection UI (before start) ---
        self.cameras = self.get_cameras_for_ui()

        values = [f"{idx} - {name}" for idx, name in self.cameras]
        default_val = values[0] if values else ""

        self.camera_var = tk.StringVar(value=default_val)

        cam_frame = ttk.LabelFrame(self.window, text="Change Camera")
        cam_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 6))

        ttk.Label(cam_frame, text="Currently used Camera:").pack(side="left", padx=(8, 4))

        self.camera_combo = ttk.Combobox(
            cam_frame,
            textvariable=self.camera_var,
            values=values,
            width=45,
            state="readonly"
        )
        self.camera_combo.pack(side="left", padx=4)

        self.start_btn = ttk.Button(cam_frame, text="Connect Camera", command=self.start_program)
        self.start_btn.pack(side="left", padx=6)


        # -------------------------
        # Kamera Setup
        # -------------------------
        self.cap = None  # wird in start_program initialisiert
        self.camera_started = False
        self._preview_cap = None
        self._preview_imgtk = None  # Referenz halten, sonst flackert’s / bleibt schwarz


        # Update interval für GUI loop
        self.update_interval = 40  # ms ≈ 25 fps GUI loop
        print(f"GUI update interval (ms): {self.update_interval}")
        
        # --- Performance throttles ---
        self._frame_i = 0
        # --- simple tic/toc profiling ---
        self._prof_enabled = True          # auf False setzen zum Abschalten
        self._prof_print_every = 25        # alle N Frames Ausgabe (25 ~ 1 Sekunde bei ~25fps)
        self._prof = {}                    # sammelt Zeiten pro Block (ms)


        # Plot nur alle n Frames aktualisieren
        self.plot_every_n = 2   # 25fps tracking, ~12.5fps plot

        # OpenCV imshow nur alle n Frames (optional)
        self.imshow_every_n = 4  # ungefähr 6.25fps Anzeige, reduziert UI-Last

        # -------------------------
        # ArUco Tracker
        # -------------------------
        tracker_cfg = ArucoTrackerConfig(
            dictionary=cv2.aruco.DICT_6X6_250,
            update_interval_ms=self.update_interval,
            max_gap_seconds=0.2,  # kurze Lücken tolerieren
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
        # ------------------------
        self.fig = Figure(figsize=(7, 6.5), dpi=100)

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
        self.ax_bar.set_ylabel("Height (cm)")
        self.ax_bar.set_xlim(-WINDOW_SECONDS, 0)
        self.ax_bar.set_ylim(0, 250)
        self.ax_bar.grid(True)

        self.fig.tight_layout()

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.right_panel)
        self.canvas.draw()
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        # -------------------------
        # Histogram Figure (Tab) (wird beim Stop befüllt)
        # -------------------------
        # Histogram plot will be created on Stop
        self.hist_fig = None
        self.hist_canvas = None
        self.ax_hist_depth = None
        self.ax_hist_xy = None

        # -------------------------
        # Bar path storage (Ring-Plot)
        # -------------------------
        self.bar_path_x = []
        self.bar_path_y = []

        # "Standpose" & "movement" thresholds (px)
        self.stand_depth_threshold_px = 10.0   # ab hier gilt: nicht mehr Standphase
        self.motion_threshold_px = 2       # minimale Änderung zwischen Frames, um als "moving" zu zählen

        # Vorwert für Bewegungsdetektion
        self._prev_bar_height_px = None

        # sauberes Beenden
        self.window.protocol("WM_DELETE_WINDOW", self.on_close)
        # Beim Start Kamera-Auswahl erzwingen (Popup)
        self.window.after(100, self.show_camera_selection_popup)


    # tic toc Profiling
    def _tic(self, key: str):
        if not getattr(self, "_prof_enabled", False):
            return None
        return time.perf_counter()

    def _toc(self, key: str, t0):
        if t0 is None:
            return
        dt_ms = (time.perf_counter() - t0) * 1000.0
        self._prof[key] = self._prof.get(key, 0.0) + dt_ms

    # Profiling Ausgabe alle N Frames
    def _prof_flush_if_needed(self):
        if not getattr(self, "_prof_enabled", False):
            return
        if (self._frame_i % self._prof_print_every) != 0:
            return
        if not self._prof:
            return
        parts = [f"{k}={v/self._prof_print_every:.1f}ms" for k, v in sorted(self._prof.items())]
        print("PERF(avg): " + " | ".join(parts))
        self._prof.clear()

    # ------------------------------------------------------------------
    # GUI Steuerung
    # ------------------------------------------------------------------
    def start_measurement(self):
        if not self.camera_started or self.cap is None:
            print("❌ Please select a camera first and click Start.")
            self.tracking_status_var.set("Tracking: ❗ select camera + Start first")
            return
        
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
            self._bar_ref_locked = False
            self._bar_ref_candidates.clear()
            self.bar_height_px = None
            self.bar_height_var.set("Bar height: -- cm")
            self._floor_marker_seen = False

            # buffers reset
            self.knee_angle_buffer.clear()
            self.bar_height_buffer.clear()

            # histogram data reset
            self.bar_path_x.clear()
            self.bar_path_y.clear()
            self._prev_bar_height_px = None

            self.is_measuring = True
            self.update_loop()
            print("Measurement started")

    def stop_measurement(self, show_hist: bool = True):
        self.is_measuring = False
        self.active_camera_var.set("Camera: --")
        print("Measurement stopped")

        if show_hist:
            self.render_histograms()
            self.notebook.select(self.tab_hist)

    def on_close(self):
        # 1) Measurement
        try:
            if getattr(self, "is_measuring", False):
                self.is_measuring = False
        except Exception:
            pass

        # 2) OpenCV Fenster schließen
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

        # 3) Kamera freigeben (falls vorhanden)
        try:
            cap = getattr(self, "cap", None)
            if cap is not None:
                try:
                    cap.release()
                except Exception:
                    pass
            self.cap = None
            self.camera_started = False
        except Exception:
            pass

        # 4) Tk sauber beenden
        try:
            self.window.quit()
        except Exception:
            pass
        try:
            self.window.destroy()
        except Exception:
            pass


    def run(self):
        self.window.mainloop()

    def list_available_cameras(self, max_index: int = 3) -> list[int]:
        available = []
        for i in range(max_index + 1):
            cap = self._open_camera_by_index(i)
            if cap is not None:
                available.append(i)
                cap.release()
        return available


    
    def get_camera_names_windows(self) -> list[str]:
        try:
            graph = FilterGraph()
            return graph.get_input_devices()
        except Exception:
            return []

    def get_cameras_for_ui(self):
        names = self.get_camera_names_windows()
        max_index = max(3, len(names) + 2)

        indices = self.list_available_cameras(max_index=max_index)  # deine open/read Prüfung
        cams = []
        for idx in indices:
            name = names[idx] if idx < len(names) else "Camera"
            cams.append((idx, name))
        return cams

    def _refresh_camera_dropdown(self):
            self.cameras = self.get_cameras_for_ui()
            values = [f"{idx} - {name}" for idx, name in self.cameras]
            self.camera_combo["values"] = values

            if values:
                # wenn aktuell ausgewählter Wert nicht mehr existiert → auf erstes setzen
                cur = self.camera_var.get().strip()
                if cur not in values:
                    self.camera_var.set(values[0])
            else:
                self.camera_var.set("")

    def show_camera_selection_popup(self):
        popup = tk.Toplevel(self.window)
        info_text = (
            "Before you start:\n"
            "• Place the laptop or camera at approximately knee height and make sure your full body is visible.\n"
            "• For best ArUco tracking, wear plain, single-color clothing with good contrast.\n\n"
            "How to begin:\n"
            "• Select the camera you want to use for the analysis (you can also switch cameras later in the program).\n"
            "• Click Connect, start the measurement, and perform your squats. 😊")
        ttk.Label(
            popup,
            text=info_text,
            justify="left", font=("Arial", 12),
            wraplength=520).pack(padx=12, pady=(0, 10), anchor="w")

        # --- Preview area ---
        preview_label = ttk.Label(popup)
        preview_label.pack(padx=12, pady=(6, 10))

        status_var = tk.StringVar(value="")
        ttk.Label(popup, textvariable=status_var).pack(padx=12, pady=(0, 8), anchor="w")

        sel0 = self.camera_var.get().strip()
        if sel0:
            cam_index0 = int(sel0.split(" - ")[0])
            ok = self._open_preview_camera(cam_index0)
            if not ok:
                status_var.set("Preview not available for this camera.")

        self._update_preview_frame(preview_label, popup, status_var)


        popup.title("Select camera")
        popup.resizable(False, False)

        # Modal: blockiert Interaktion mit Hauptfenster
        popup.transient(self.window)
        popup.grab_set()

        # Inhalt
        ttk.Label(popup, text="Select the prefered camera for squat analysis:", font=("Arial", 12)).pack(
            padx=12, pady=(12, 8)
        )

        # Dropdown
        self._refresh_camera_dropdown()
        values = list(self.camera_combo["values"])
        if values:
            self.camera_var.set(values[0])
        else:
            self.camera_var.set("")

        combo = ttk.Combobox(
            popup,
            textvariable=self.camera_var,
            values=values,
            width=45,
            state="readonly",
        )
        combo.pack(padx=12, pady=(0, 10))

        # Kamera-Wechsel Event
        def on_cam_change(_evt=None):
            sel = self.camera_var.get().strip()
            if not sel:
                return
            cam_index = int(sel.split(" - ")[0])
            ok = self._open_preview_camera(cam_index)
            if not ok:
                status_var.set("Preview not available for this camera.")

        combo.bind("<<ComboboxSelected>>", on_cam_change)

        # Status
        status_var = tk.StringVar(value="")
        ttk.Label(popup, textvariable=status_var).pack(padx=12, pady=(0, 8))

        # Button row
        btn_row = ttk.Frame(popup)
        btn_row.pack(padx=12, pady=(0, 12), fill="x")

        def do_connect():
            if not self.camera_var.get().strip():
                status_var.set("No camera found.")
                return
            self._close_preview()
            self.start_program()  # öffnet Kamera anhand camera_var
            if getattr(self, "camera_started", False):
                popup.destroy()
            else:
                status_var.set("Could not open selected camera. Please choose another one.")

        def do_exit():
            # User will nicht wählen -> App schließen
            popup.destroy()
            self._close_preview()
            popup.destroy()
            self.on_close()

        ttk.Button(btn_row, text="Connect", command=do_connect).pack(side="left")
        ttk.Button(btn_row, text="Exit", command=do_exit).pack(side="right")

        # Popup zentrieren (optional, aber nice)
        self.window.update_idletasks()
        x = self.window.winfo_rootx() + 60
        y = self.window.winfo_rooty() + 60
        popup.geometry(f"+{x}+{y}")

        # WICHTIG: Wenn User auf X am Popup drückt -> App schließen (oder du kannst es blockieren)
        popup.protocol("WM_DELETE_WINDOW", do_exit)


    def start_program(self):
        sel = self.camera_var.get().strip()
        if not sel:
            print("❌ No camera selected.")
            self.camera_started = False
            self.tracking_status_var.set("Tracking: ❌ no camera selected")
            return
        
        self.camera_started = True
        self.tracking_status_var.set(f"Tracking: camera {sel} ready ✅")
        self.active_camera_var.set(f"Camera: {sel}")
        print(f"✅ Camera opened: {sel}")

        # falls Messung läuft: sauber stoppen
        if self.is_measuring:
            self.stop_measurement(show_hist=False)
            self.notebook.select(self.tab_live)  

        # OpenCV-Fenster schließen (wichtig beim Wechsel)
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

        # alte Kamera freigeben
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None

        # kurze Pause, damit Windows den Device-Handle freigibt
        import time
        time.sleep(0.2)

        # Kameraindex aus UI
        sel = self.camera_var.get().strip()
        if not sel:
            print("❌ No camera selected.")
            self.camera_started = False
            self.tracking_status_var.set("Tracking: ❌ no camera selected")
            return

        cam_index = int(sel.split(" - ")[0])


        # Falls schon offen -> sauber schließen
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None

        # Backend (Windows: DSHOW bewährt). Wenn du nicht Windows nutzt: einfach cv2.VideoCapture(cam_index)
        cap = self._open_camera_by_index(cam_index)
        if cap is None:
            print(f"❌ Cannot open camera index {cam_index}")
            self.cap = None
            self.camera_started = False
            self.tracking_status_var.set("Tracking: ❌ camera not opened")
            return
        self.cap = cap

        if not self.cap.isOpened():
            print(f"❌ Cannot open camera index {cam_index}")
            self.cap = None
            self.camera_started = False
            self.tracking_status_var.set("Tracking: ❌ camera not opened")
            return

        # Low-latency / Performance settings (best effort)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        

        if not self.cap.isOpened():
            print(f"❌ Cannot open camera index {cam_index}")
            self.camera_started = False
            self.tracking_status_var.set("Tracking: ❌ camera not opened")
            return

        self.camera_started = True
        self.tracking_status_var.set(f"Tracking: camera {cam_index} ready ✅")
        print(f"✅ Camera {cam_index} opened successfully")

    def _open_camera_by_index(self, index: int):
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                return cap
        cap.release()
        return None


    # ------------------------------------------------------------------
    # Main Loop
    # ------------------------------------------------------------------
    def update_loop(self):
        # Kamera prüfen
        if self.cap is None or (not self.cap.isOpened()):
            self.tracking_status_var.set("Tracking: ❌ camera not available")
            self.stop_measurement()
            return

        if not self.is_measuring:
            return

        try:
            # Profiling reset
            t0 = self._tic("cap.read")
            ret, frame = self.cap.read()
            self._toc("cap.read", t0)

            if not ret:
                print("Failed to grab frame")
                self.window.after(100, self.update_loop)
                return

            t0 = self._tic("process_frame")
            femur_angle, knee_valid, squat_depth_angle, bar_height_px, bar_height_abs_cm = self.process_frame(frame)
            self._toc("process_frame", t0)


            # Tracking Status
            self.update_tracking_status(self.last_centers)

            # GUI Winkel
            self.femur_angle_var.set(f"Femur angle: {femur_angle:.1f} °" if femur_angle is not None else "Femur angle: -- °")
            self.knee_valid_angle_var.set(f"Knee angle: {knee_valid:.1f} °" if knee_valid is not None else "Knee angle: -- °")
            # Bar Höhe in cm (wenn Kalibrierung da)
            mm_per_px = self.tracker.mm_per_px
            if bar_height_abs_cm is not None:
                self.bar_height_var.set(f"Bar height: {bar_height_abs_cm:.1f} cm")
            else:
                self.bar_height_var.set("Bar height: -- cm")


            # --- Stabilisierung + Statusanzeige + Counter ---
            cls, ok_stable, high_stable = self.update_validity_stability(squat_depth_angle, knee_valid)

            # Tracking status text (für Overlay)
            tracking_text = self.tracking_status_var.get() if hasattr(self, "tracking_status_var") else None

            # progress: 0..1 (0=zu hoch, 1=valid tief)
            progress = None
            if squat_depth_angle is not None:
                # high_threshold ~ oben, ok_threshold ~ unten (tiefer)
                top = self.depth_high_threshold       # z.B. +2
                bottom = self.depth_ok_threshold      # z.B. -2
                if top != bottom:
                    progress = (top - squat_depth_angle) / (top - bottom)
                    progress = max(0.0, min(1.0, float(progress)))

            t0 = self._tic("cv_draw")
            # Live-Farbfeedback ins Kamerabild
            self.draw_live_feedback(frame, cls, tracking_text=tracking_text, progress=progress)
            self._toc("cv_draw", t0)

            # Cooldown runterzählen
            if self._cooldown_frames > 0:
                self._cooldown_frames -= 1

            # Kombinierte Stabilisierung (Depth + Knee_valid)
            #cls, ok_stable, high_stable = self.update_validity_stability(squat_depth_angle, knee_valid)

            # Anzeige
            if cls is None:
                self.squat_status_var.set("Squat Status: --")
            elif cls == "OK":
                self.squat_status_var.set("Squat Status: ✅ depth OK")
            elif cls == "HIGH":
                self.squat_status_var.set("Squat Status: ❌ too high")
            else:
                self.squat_status_var.set("Squat Status: ⚠ borderline")

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

            mm_per_px = self.tracker.mm_per_px
            if bar_height_abs_cm is not None:
                self.bar_height_buffer.append(bar_height_abs_cm)
            else:
                self.bar_height_buffer.append(float("nan"))

            # --- Live Plots updaten (throttled) ---
            self._frame_i += 1
            if (self._frame_i % self.plot_every_n) == 0:
                t0 = self._tic("plot")
                self.update_live_plots()
                self._toc("plot", t0)

            if (self.MARKER_BAR_ID in self.last_centers) and (bar_height_px is not None):
                bar = self.last_centers[self.MARKER_BAR_ID]
                bar_x_px = float(bar[0])
                bar_y_px = float(bar[1])

                mm_per_px = self.tracker.mm_per_px
                if mm_per_px is not None:
                    bar_x_cm = (bar_x_px * mm_per_px) / 10.0

                    # Bewegung erkennen weiterhin in px
                    moving = False
                    if (bar_height_px is not None) and (self._prev_bar_height_px is not None):
                        if abs(bar_height_px - self._prev_bar_height_px) >= self.motion_threshold_px:
                            moving = True
                    self._prev_bar_height_px = bar_height_px

                    # Nur Bewegungsphase sammeln (Standphase raus)
                    if (cls is not None) and (bar_height_px > self.stand_depth_threshold_px) and moving:
                        if bar_height_abs_cm is not None:
                            self.bar_path_x.append(bar_x_cm)
                            self.bar_path_y.append(bar_height_abs_cm)

                else:
                    # Keine Kalibrierung -> nichts sammeln (sonst px als cm gelabelt)
                    self._prev_bar_height_px = bar_height_px

            # Boden-Referenzlinie
            h, w, _ = frame.shape
            y_floor = int(h * 0.8)
            t0 = self._tic("cv_draw")
            cv2.line(frame, (0, y_floor), (w, y_floor), (0, 255, 0), 2)

            t0 = self._tic("cv_draw")
            # Segmente + Bar Marker zeichnen
            self.draw_segments(frame)
            self._toc("cv_draw", t0)

            t0 = self._tic("cv_draw")
            # Overlay Status-Text
            cv2.putText(frame, f"State: {self._squat_state}", (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            self._toc("cv_draw", t0)

            # Frame anzeigen
            t0 = self._tic("imshow")
            if (self._frame_i % self.imshow_every_n) == 0:
                cv2.imshow("Squat Camera", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    self.stop_measurement()
            else:
                # trotzdem Events pumpen, minimal
                cv2.waitKey(1)
            self._toc("imshow", t0)

            # Profiling Ausgabe wenn gewünscht
            self._prof_flush_if_needed()
            self.window.after(self.update_interval, self.update_loop)

        except Exception as e:
            print(f"❌ Error in update_loop: {e}")
            self.squat_status_var.set("Squat: ERROR (see console)")
            self.stop_measurement()

    # ------------------------------------------------------------------
    # Process frame: marker -> angles + bar height
    # ------------------------------------------------------------------
    def process_frame(self, frame):
        # Start: nehme die letzten Zentren als Basis
        centers = dict(self.last_centers) if self.last_centers else {}

        try:
            t0 = self._tic("aruco.update")
            detected = self.tracker.update(frame, draw=True)
            self._toc("aruco.update", t0)
            # Merge: neue Detektionen überschreiben alte, aber fehlende bleiben kurz erhalten
            if detected is not None and len(detected) > 0:
                centers.update(detected)
        except Exception as e:
            print("Tracker error:", e)
            # centers bleibt dann die letzte brauchbare Schätzung

        self.last_centers = centers

        t0 = self._tic("angles")
        femur_angle = femur_angle_depth_signed_deg(centers, self.MARKER_HIP_ID, self.MARKER_KNEE_ID)
        depth_angle = squat_depth_angle_deg(centers, self.MARKER_HIP_ID, self.MARKER_KNEE_ID)
        knee_valid = knee_valid_angle_deg(centers, self.MARKER_HIP_ID, self.MARKER_KNEE_ID, self.MARKER_ANKLE_ID, reference_deg=90.0)
        self._toc("angles", t0)

        # --- Bar depth (relative, px) ---
        bar_height_px = None
        bar_y_px = None

        # --- Absolute bar height above floor (cm) ---
        bar_height_abs_cm = None
        MARKER_SIZE_MM = 90.0  # Boden-ArUco ist 90x90 mm

        if self.MARKER_BAR_ID in centers:
            bar_y_px = float(centers[self.MARKER_BAR_ID][1])

            # --- set bar reference (lock when still) ---
            if not self._bar_ref_locked:
                self._bar_ref_candidates.append(bar_y_px)

                if len(self._bar_ref_candidates) >= self._bar_ref_min_samples:
                    y_min = min(self._bar_ref_candidates)
                    y_max = max(self._bar_ref_candidates)
                    y_range = y_max - y_min

                    if y_range <= self._bar_still_range_px:
                        self.bar_y_ref = float(np.median(self._bar_ref_candidates))
                        self._bar_ref_locked = True

            # relative displacement (für movement / standphase)
            if self.bar_y_ref is not None:
                bar_height_px = bar_y_px - self.bar_y_ref

            # absolute height using FLOOR marker
            mm_per_px = self.tracker.mm_per_px
            if (mm_per_px is not None) and (self.MARKER_FLOOR_ID in centers):
                self._floor_marker_seen = True
                floor_center_y_px = float(centers[self.MARKER_FLOOR_ID][1])

                marker_size_px = MARKER_SIZE_MM / mm_per_px
                floor_ground_y_px = floor_center_y_px + 0.5 * marker_size_px

                bar_height_abs_cm = (floor_ground_y_px - bar_y_px) * mm_per_px / 10.0

        self.bar_height_px = bar_height_px
        return femur_angle, knee_valid, depth_angle, bar_height_px, bar_height_abs_cm

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
            pad = max(1.0, 0.15 * (vmax - vmin + 1e-9))
            low = max(0.0, vmin - pad)             # depth should not go far below 0
            high = vmax + pad
            # if you're barely moving, avoid micro-scale like 5 cm
            if high - low < 5:
                high = low + 5
            self.ax_bar.set_ylim(low, high)
        else:
            self.ax_bar.set_ylim(0, 50)

        self.canvas.draw_idle()
    
    def update_validity_stability(self, squat_depth_angle, knee_valid):
        """
        Kombinierte Stabilisierung:
        - squat_depth_angle <= 0
        - knee_valid <= 0
        """

        if squat_depth_angle is None or knee_valid is None:
            self._ok_streak = 0
            self._high_streak = 0
            return None, False, False

        # gültig
        if squat_depth_angle <= 0 and knee_valid <= 0:
            self._ok_streak += 1
            self._high_streak = 0
        else:
            self._high_streak += 1
            self._ok_streak = 0

        ok_stable = self._ok_streak >= self.stable_frames_required
        high_stable = self._high_streak >= self.stable_frames_required

        if ok_stable:
            cls = "OK"
        elif high_stable:
            cls = "HIGH"
        else:
            cls = "TRANSITION"

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
            self.MARKER_FLOOR_ID: "FLOOR",
        }

        present = [mid for mid in required.keys() if mid in centers]
        missing = [required[mid] for mid in required.keys() if mid not in centers]

        if len(present) == 5:
            status = "EXCELLENT"
        elif len(present) == 4:
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
            self.tracking_status_var.set("Tracking: EXCELLENT (5/5 markers)")

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

    # Klasse zur Stabilitätsbewertung der Squat-Tiefe
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

    def draw_live_feedback(self, frame, cls, tracking_text=None, progress=None):
        """
        Draws a colored status box + text overlay on the camera frame.
        cls: None / "OK" / "HIGH" / "TRANSITION" / "BORDERLINE"
        """

        # Farben (BGR!)
        if cls == "OK":
            color = (0, 200, 0)       # grün
            text = "VALID"
        elif cls == "HIGH":
            color = (0, 0, 255)       # rot
            text = "TOO HIGH"
        elif cls in ("TRANSITION", "BORDERLINE"):
            color = (0, 215, 255)     # gelb/orange
            text = "BORDERLINE"
        else:
            color = (180, 0, 180)     # violett
            text = "TRACKING LOST"

        h, w = frame.shape[:2]

        # Hintergrundbox (oben links)
        overlay = frame.copy()
        x1, y1 = 10, 10
        x2, y2 = 310, 80
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        alpha = 0.35  # 0.0 = unsichtbar, 1.0 = voll deckend
        frame[:] = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)


        # Text in die Box (weiß)
        cv2.putText(frame, text, (x1 + 10, y1 + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

        # Optional: Tracking-Text darunter
        #if tracking_text:
            #cv2.putText(frame, tracking_text, (10, 95),
                        #cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        # Rahmen ums ganze Bild (starker visueller Hinweis)
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), color, 6)

        
        #  vertical progress bar (right side)
        # progress: 0..1 (0=zu hoch, 1=valid tief). Falls None -> nur Rahmen anzeigen
        # Tipp: Übergib progress aus update_loop (wie besprochen). Wenn du’s noch nicht tust:
        # progress = None lassen oder erstmal fix 0.5 testen.

        bar_x = x1                 # links bündig mit Box
        bar_y = y2 + 12            # unter Box
        bar_w = 36                 # schlank/modern
        bar_h = 160                # Höhe der Bar

        # falls Bild klein ist -> Bar nach oben begrenzen
        bar_y2 = min(bar_y + bar_h, h - 10)
        bar_y1 = bar_y
        bar_h_eff = bar_y2 - bar_y1
        bar_x1 = bar_x
        bar_x2 = bar_x1 + bar_w

        # Halbtransparenter Hintergrund
        overlay = frame.copy()
        cv2.rectangle(overlay, (bar_x1, bar_y1), (bar_x2, bar_y2), (0, 0, 0), -1)
        frame[:] = cv2.addWeighted(overlay, 0.28, frame, 0.72, 0)

        # Dezenter Rahmen
        cv2.rectangle(frame, (bar_x1, bar_y1), (bar_x2, bar_y2), (255, 255, 255), 1, cv2.LINE_AA)

        # "Ticks" für Orientierung (HIGH / BORDERLINE / OK)
        y_tick1 = bar_y1 + int(bar_h_eff * 0.33)
        y_tick2 = bar_y1 + int(bar_h_eff * 0.66)
        cv2.line(frame, (bar_x1 + 5, y_tick1), (bar_x2 - 5, y_tick1), (255, 255, 255), 1, cv2.LINE_AA)
        cv2.line(frame, (bar_x1 + 5, y_tick2), (bar_x2 - 5, y_tick2), (255, 255, 255), 1, cv2.LINE_AA)

        # "Rounded" Look
        pad = 4
        ix1, iy1 = bar_x1 + pad, bar_y1 + pad
        ix2, iy2 = bar_x2 - pad, bar_y2 - pad

        #mInnenfläche leicht abdunkeln (gibt depth)
        overlay2 = frame.copy()
        cv2.rectangle(overlay2, (ix1, iy1), (ix2, iy2), (0, 0, 0), -1)
        frame[:] = cv2.addWeighted(overlay2, 0.12, frame, 0.88, 0)

        # Füllstand zeichnen
        # progress in [0..1], 0=oben (zu hoch), 1=unten (valid tief)
        if progress is not None and bar_h_eff > 8:
            p = max(0.0, min(1.0, float(progress)))
            fill_top = iy1 + int((1.0 - p) * (iy2 - iy1))
            fill_top = max(iy1, min(fill_top, iy2))

            # Füllung
            cv2.rectangle(frame, (ix1, fill_top), (ix2, iy2), color, -1)

            # "Rounded caps" simulieren: kleine Kreise oben/unten der Füllung
            cx = (ix1 + ix2) // 2
            r = max(2, (ix2 - ix1) // 2)
            cv2.circle(frame, (cx, iy2), r, color, -1)          # unten rund
            cv2.circle(frame, (cx, fill_top), r, color, -1)     # oben rund

        # Labels rechts daneben
        cv2.putText(frame, "HIGH", (bar_x2 + 10, bar_y1 + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, "OK", (bar_x2 + 10, bar_y2 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    
    # Histogram rendering (called on stop)
    def render_histograms(self):
        # --- old canvas entfernen (falls vorhanden) ---
        if self.hist_canvas is not None:
            self.hist_canvas.get_tk_widget().destroy()
            self.hist_canvas = None
            self.hist_fig = None

        # --- neue Figure erstellen ---
        self.hist_fig = Figure(figsize=(8.0, 5.0), dpi=100)

        # Joint-Layout: oben Histogramm-x, rechts Histogramm-y, Mitte Ring
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

        # --- Check data ---
        if not self._floor_marker_seen:
            ax_main.text(
                0.5, 0.5,
                "No histogram data created.\n"
                "Floor marker (ID 44) was not detected.",
                ha="center", va="center",
                fontsize=12,
                transform=ax_main.transAxes
            )
            ax_main.set_axis_off()

        elif len(self.bar_path_x) < 10:
            ax_main.text(
                0.5, 0.5,
                "Not enough MOVEMENT data collected.\n"
                "Check thresholds or movement execution.",
                ha="center", va="center",
                fontsize=12,
                transform=ax_main.transAxes
            )
            ax_main.set_axis_off()

        else:
            x = np.asarray(self.bar_path_x, dtype=float)
            y = np.asarray(self.bar_path_y, dtype=float)

            # Main: 2D density + trajectory line (Ring)
            ax_main.hist2d(x, y, bins=30)
            ax_main.plot(x, y, linewidth=1)  # Trajektorie darüber

            ax_main.set_title("Bar path during squats (movement phases)")
            ax_main.set_xlabel("Horizontal Movement (cm)")
            ax_main.set_ylabel("Height above floor (cm)")
            ax_main.grid(True)

            # Marginals: x oben, y rechts
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

            # damit das rechte Histogramm "nach innen" zeigt
            ax_histy.invert_xaxis()

        # --- Canvas in Histogram-Tab einbetten ---
        self.hist_canvas = FigureCanvasTkAgg(self.hist_fig, master=self.tab_hist)
        self.hist_canvas.draw()
        self.hist_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, pady=10)

    # ------------------------------------------------------------------
    # Sound
    # ------------------------------------------------------------------
    def play_valid_squat_sound(self):
        _play_valid_squat_sound(self.sound_enabled_var.get())

    def _open_preview_camera(self, cam_index: int):
        # alte Preview schließen
        if self._preview_cap is not None:
            try:
                self._preview_cap.release()
            except Exception:
                pass
            self._preview_cap = None

        # Preview
        cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)

        if not cap.isOpened():
            cap.release()
            return False

        # settings for low-latency
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # warm-up Frames verwerfen
        ok = False
        for _ in range(3):
            ret, _ = cap.read()
            if ret:
                ok = True
                break

        if not ok:
            cap.release()
            return False

        self._preview_cap = cap
        return True

    
    def _update_preview_frame(self, preview_label, popup, status_var):
        # Popup wurde geschlossen?
        try:
            if not popup.winfo_exists():
                return
        except Exception:
            return

        if self._preview_cap is None:
            status_var.set("No preview available.")
            popup.after(200, lambda: self._update_preview_frame(preview_label, popup, status_var))
            return

        ret, frame = self._preview_cap.read()
        if not ret or frame is None:
            status_var.set("Preview: failed to grab frame.")
            popup.after(200, lambda: self._update_preview_frame(preview_label, popup, status_var))
            return

        status_var.set("")  # ok

        # BGR -> RGB -> ImageTk
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)

        # optional: kleiner anzeigen (schneller, passt ins Popup)
        img.thumbnail((520, 320))

        self._preview_imgtk = ImageTk.PhotoImage(img)
        preview_label.configure(image=self._preview_imgtk)

        popup.after(30, lambda: self._update_preview_frame(preview_label, popup, status_var))

    def _close_preview(self):
        if self._preview_cap is not None:
            try:
                self._preview_cap.release()
            except Exception:
                pass
            self._preview_cap = None
        self._preview_imgtk = None



