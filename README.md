# Car License Plate Detection

This project provides a real-time, YOLO-based license plate detection system. You can run the detection simply via a command-line interface or use an intuitive graphical user interface (GUI) to configure and run the model on video files or your webcam.

## Features

- **Real-Time Detection:** Rapidly identify license plates using the advanced YOLO object detection architecture.
- **Graphical User Interface:** Easy-to-use GUI built with `Tkinter` to dynamic-modify thresholds, input sources, and resolutions.
- **CLI Support:** Run standard detections directly from the command line constraints via `main.py`.
- **Dynamic Configuration:** Adjust Confidence, NMS IoU, and Inference image size on the fly while detection is running (via the GUI).
- **Save Outputs:** Optionally toggle Saving output frames to an `.mp4` file (`realtime_detection_output.mp4`).
- **Jupyter Notebook included:** `Training.ipynb` included for custom model training procedures.

## Requirements

The core functionality relies on `python >= 3.8`. Ensure you have installed the required libraries by running:

```bash
pip install -r requirements.txt
```
*(Note: `tkinter` is built-in with most Python installations on Windows. If you are on Linux, you might need to install `python3-tk`)*

## Usage

You must provide a trained YOLO model. By default, both interfaces look for a `best.pt` file in the root directory.

### 1. Using the GUI (Recommended)

Run the intuitive interface to configure options manually and get real-time previews:

```bash
python gui.py
```

**GUI Options:**
- **Model Path:** Path to your `.pt` YOLO weights file.
- **Video Source:** Default is `0` for the primary webcam. You can also click "Browse" or type in the path to a video file.
- **Confidence:** Slider measuring the minimum threshold to approve a detection.
- **NMS IoU:** Intersection over Union threshold used in Non-Maximum Suppression to filter duplicate boxes.
- **Inference Size:** YOLO model resolution grid. Higher resolutions improve small plate detection at a cost to FPS.

### 2. Using the CLI

If you prefer command-line execution or batch-script integration, use `main.py`:

```bash
python main.py --model best.pt --source 0 --conf 0.35 --iou 0.45 --imgsz 640
```

**Arguments:**
- `--model`: Path to YOLO weights (defaults to searching for `best.pt`).
- `--source`: Webcam index (like `0`) or video file path.
- `--conf`: Confidence threshold (0-1).
- `--iou`: NMS IoU threshold (0-1).
- `--imgsz`: Inference image size (int).
- `--hide-fps`: Disables the FPS counter overlay.
- `--save-output`: Will save the outcome to an output video recording.
- `--output`: Filepath for saved output (default: `realtime_detection_output.mp4`).

### 3. Training a Custom Model

If you want to train your own YOLO model from scratch or fine-tune an existing one on a custom license plate dataset, this project includes a dedicated Jupyter Notebook to help you do so:

- **`Training.ipynb`**: This notebook serves as an interactive walkthrough for model training. It contains the code required to load your datasets, train the YOLO model using the `ultralytics` package, validate the model's accuracy, and export the resulting weights (which you can then load back into `gui.py` or `main.py`).

To get started with training, ensure you have installed the requirements, then launch Jupyter and open the notebook:
```bash
jupyter notebook Training.ipynb
```

## Stopping Execution

During detection (in both CLI and GUI), press the **`q`** key or close the OpenCV display window to safely stop the stream and save output files tracking cleanly.