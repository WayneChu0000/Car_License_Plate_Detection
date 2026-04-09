import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import os
from pathlib import Path

# Import the core logic from ai.py
from ai import run_realtime_plate_detection, resolve_model_path

class LicensePlateGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Real-Time License Plate Detection")
        self.root.geometry("500x550")
        self.root.resizable(False, False)
        
        # Stop event for the detection thread
        self.stop_event = threading.Event()
        # Dictionary to pass parameters dynamically to the AI module
        self.shared_config = {}

        # Handle window close to ensure clean exit
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Apply modern styling
        style = ttk.Style(self.root)
        style.theme_use('clam')
        
        self.create_widgets()
        self.update_shared_config() # Initialize the dictionary

    def create_widgets(self):
        # Main Frame
        main_frame = ttk.Frame(self.root, padding="20 20 20 20")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Title
        title_label = ttk.Label(
            main_frame, 
            text="License Plate Detection Configuration", 
            font=("Helvetica", 14, "bold")
        )
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))

        # 1. Model Selection
        ttk.Label(main_frame, text="Model Path (.pt):", font=("Helvetica", 10)).grid(row=1, column=0, sticky=tk.W, pady=5)
        self.model_var = tk.StringVar(value="best.pt")
        ttk.Entry(main_frame, textvariable=self.model_var, width=30).grid(row=1, column=1, padx=5, pady=5)
        ttk.Button(main_frame, text="Browse", command=self.browse_model).grid(row=1, column=2, pady=5)

        # 2. Video Source
        ttk.Label(main_frame, text="Video Source:", font=("Helvetica", 10)).grid(row=2, column=0, sticky=tk.W, pady=5)
        self.source_var = tk.StringVar(value="0")
        ttk.Entry(main_frame, textvariable=self.source_var, width=30).grid(row=2, column=1, padx=5, pady=5)
        ttk.Button(main_frame, text="Browse", command=self.browse_source).grid(row=2, column=2, pady=5)
        ttk.Label(main_frame, text="(Enter '0' for webcam, or path to video file)", font=("Helvetica", 8)).grid(row=3, column=1, sticky=tk.W, pady=(0, 10))

        # 3. Confidence Threshold
        ttk.Label(main_frame, text="Confidence (0-1):", font=("Helvetica", 10)).grid(row=4, column=0, sticky=tk.W, pady=5)
        self.conf_var = tk.DoubleVar(value=0.35)
        conf_slider = ttk.Scale(main_frame, from_=0.01, to=1.0, variable=self.conf_var, orient=tk.HORIZONTAL, command=self.update_conf_label)
        conf_slider.grid(row=4, column=1, sticky=tk.EW, padx=5, pady=5)
        self.conf_label = ttk.Label(main_frame, text="0.35")
        self.conf_label.grid(row=4, column=2, pady=5)

        # 4. IoU Threshold
        ttk.Label(main_frame, text="NMS IoU (0-1):", font=("Helvetica", 10)).grid(row=5, column=0, sticky=tk.W, pady=5)
        self.iou_var = tk.DoubleVar(value=0.45)
        iou_slider = ttk.Scale(main_frame, from_=0.01, to=1.0, variable=self.iou_var, orient=tk.HORIZONTAL, command=self.update_iou_label)
        iou_slider.grid(row=5, column=1, sticky=tk.EW, padx=5, pady=5)
        self.iou_label = ttk.Label(main_frame, text="0.45")
        self.iou_label.grid(row=5, column=2, pady=5)

        # 5. Image Size
        ttk.Label(main_frame, text="Inference Size:", font=("Helvetica", 10)).grid(row=6, column=0, sticky=tk.W, pady=5)
        self.imgsz_var = tk.StringVar(value="640")
        imgsz_combo = ttk.Combobox(main_frame, textvariable=self.imgsz_var, values=["320", "480", "640", "1280"], state="readonly", width=10)
        imgsz_combo.grid(row=6, column=1, sticky=tk.W, padx=5, pady=5)
        self.imgsz_var.trace_add("write", self.update_shared_config)

        # 6. Target Output FPS (Optional override)
        ttk.Label(main_frame, text="Output Video FPS:", font=("Helvetica", 10)).grid(row=7, column=0, sticky=tk.W, pady=5)
        self.target_fps_var = tk.StringVar(value="Auto")
        fps_combo = ttk.Combobox(main_frame, textvariable=self.target_fps_var, values=["Auto", "15", "24", "30", "60"], state="readonly", width=10)
        fps_combo.grid(row=7, column=1, sticky=tk.W, padx=5, pady=5)

        # 7. Checkboxes (Show FPS, Save Output)
        self.show_fps_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(main_frame, text="Show Current FPS Target Overlay", variable=self.show_fps_var, command=self.update_shared_config).grid(row=8, column=0, columnspan=2, sticky=tk.W, pady=5)

        self.save_video_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(main_frame, text="Save Output to realtime_detection_output.mp4", variable=self.save_video_var).grid(row=9, column=0, columnspan=3, sticky=tk.W, pady=5)

        # 8. Start/Stop Button
        self.start_btn = ttk.Button(main_frame, text="▶ Start Detection", command=self.start_detection)
        self.start_btn.grid(row=10, column=0, columnspan=3, pady=25, ipadx=10, ipady=10)

    def update_conf_label(self, val):
        self.conf_label.config(text=f"{float(val):.2f}")
        self.update_shared_config()

    def update_iou_label(self, val):
        self.iou_label.config(text=f"{float(val):.2f}")
        self.update_shared_config()

    def on_closing(self):
        """Called when the user clicks 'X' to close the GUI."""
        self.stop_event.set()
        self.root.destroy()
        os._exit(0)  # Force complete exit of all threads to prevent OpenCV hangs

    def update_shared_config(self, *args):
        """Updates the shared config dict used by the YOLO loop dynamically."""
        try:
            self.shared_config['conf'] = self.conf_var.get()
            self.shared_config['iou'] = self.iou_var.get()
            self.shared_config['imgsz'] = int(self.imgsz_var.get())
            self.shared_config['show_fps'] = self.show_fps_var.get()
        except:
            pass  # Values might not be initialized yet when this is first called

    def browse_model(self):
        filepath = filedialog.askopenfilename(
            title="Select YOLO Model",
            filetypes=(("PyTorch Models", "*.pt"), ("All Files", "*.*"))
        )
        if filepath:
            self.model_var.set(filepath)

    def browse_source(self):
        filepath = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=(("MP4 files", "*.mp4"), ("AVI files", "*.avi"), ("All Files", "*.*"))
        )
        if filepath:
            self.source_var.set(filepath)

    def start_detection(self):
        # Gather inputs
        model_path = self.model_var.get()
        source_raw = self.source_var.get()
        conf = self.conf_var.get()
        iou = self.iou_var.get()
        imgsz = int(self.imgsz_var.get())
        show_fps = self.show_fps_var.get()
        save_output = self.save_video_var.get()
        
        target_fps_raw = self.target_fps_var.get()
        target_fps = int(target_fps_raw) if target_fps_raw.isdigit() else None

        try:
            resolved_model = resolve_model_path(model_path)
        except Exception as e:
            messagebox.showerror("Error", str(e))
            return

        source = int(source_raw) if source_raw.isdigit() else source_raw

        # Disable button while running
        self.start_btn.config(state=tk.DISABLED, text="Running... (Close OpenCV Window to stop)")
        self.stop_event.clear()

        def run_thread():
            try:
                run_realtime_plate_detection(
                    model_path=resolved_model,
                    video_source=source,
                    conf=conf,
                    iou=iou,
                    imgsz=imgsz,
                    show_fps=show_fps,
                    save_output=save_output,
                    target_fps=target_fps,
                    config_dict=self.shared_config,
                    stop_event=self.stop_event
                )
            except Exception as e:
                messagebox.showerror("Execution Error", str(e))
            finally:
                # Re-enable button
                self.start_btn.config(state=tk.NORMAL, text="▶ Start Detection")
                self.stop_event.clear()

        threading.Thread(target=run_thread, daemon=True).start()

if __name__ == "__main__":
    root = tk.Tk()
    app = LicensePlateGUI(root)
    root.mainloop()
