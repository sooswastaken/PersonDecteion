# Quick-Start Person Detection & Tracking Pipeline

This repository contains a minimal, end-to-end pipeline that lets you:

1. **Calibrate** a ceiling-mounted camera using a printed checkerboard.
2. **Compute a floor homography** so pixel coordinates map to real-world  (X,Y) positions.
3. **Detect & track people** in live video (YOLOv5n → SORT) and export their 2-D floor locations.
4. **Fuse multiple cameras** (optional) so two or more views share a single set of track IDs and (X,Y) coordinates.

---

## Hardware Setup

* Web camera mounted ~2.5-3 m above the floor, angled downward.
* Floor area of interest ~5 × 5 m visible in the image.
* Optional Raspberry Pi 4 + Coral USB accelerator works (the script defaults to CPU).

---

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Ensure **PyTorch** matches your system/CUDA version (the requirement here installs CPU-only by default).

---

## Usage

### 1. Camera Calibration

Print a **9 × 6** inner-corner checkerboard (each square side = `SQUARE_SIZE` in `quick_start_pipeline.py`, default 2.5 cm).

```bash
python quick_start_pipeline.py calibrate --cam 0 --frames 20
```

* Move the checkerboard around the floor, pressing **`c`** when fully visible.
* Collect ≥ 5 good frames (default 20 gives better accuracy).
* Calibration parameters are saved to `camera_intrinsics.npz`.

### 2. Floor Homography

After calibration, run:

```bash
python quick_start_pipeline.py homography --cam 0
```

* A live frame appears (undistorted). Click **four** floor points **clockwise** (e.g. tape an “X” at each room corner).
* After clicking, enter the real-world **X Y** coordinates (meters) for each point.
* Homography is saved to `floor_homography.npy`.

### 3. Detection & Tracking

```bash
python quick_start_pipeline.py detect --cam 0          # with SORT tracking
python quick_start_pipeline.py detect --cam 0 --no-track  # detection only
```

* Bounding boxes are drawn in green.
* Each ID’s current ground-plane position is overlaid as `ID n @ (X,Y) m`.
* Press **`q`** to quit.

### 4. Multi-Camera Fusion (optional)

After you’ve calibrated and homography-mapped **each** camera, run:

```bash
python run_all.py --cams 0 1            # two cameras
python run_all.py --cams 0 1 2 --display  # three cams + live windows
```

`run_all.py` spawns one detector worker per camera (implemented in `detect_cam.py`) and a central **FusionCenter** that merges detections within a configurable distance gate.

Arguments:
* `--gating 0.75` — metre radius to treat detections as the same person (default 0.75 m)
* `--max-age 1.0` — drop a track if no camera has seen it for this many seconds

Console output shows active track IDs and positions. Extend or redirect this easily to WebSockets, MQTT, etc.

> **Tip** If your cameras are numbered differently (e.g. /dev/video2, /dev/video5), just pass the indices: `--cams 2 5`.

---

## Customisation

* **Different Model** – Replace `torch.hub.load("ultralytics/yolov5", "yolov5n", ...)` with any YOLOv5/YOLOv8 variant or another detector.
* **Tracker** – If the `sort` package isn’t available, the script falls back to detection-only mode.
* **Checkerboard Size** – Change `CHECKERBOARD` and `SQUARE_SIZE` constants.

---

## Troubleshooting

* *ModuleNotFoundError: torch* – Install the correct PyTorch wheel for your platform/CUDA.
* *Camera not opening* – Try another `--cam` index or check OS permissions.
* *Few detections* – Increase ambient lighting, move the camera closer, or try a larger model (`yolov5s`).

---

## License

This project is released under the MIT License. All third-party models and libraries retain their original licenses. 

---

## Multi-Camera Mode

Want wider coverage or fewer occlusions? Use `run_all.py` to spawn one detector process per camera and fuse them into a single set of world-space tracks.

```bash
python run_all.py --cams 0 1        # two cameras (indices 0 and 1)
python run_all.py --cams 2 3 --display  # show live views, press q to quit a window
```

How it works:
1. **Each worker** (`detect_cam.camera_worker`) detects people, projects box bottoms to floor  (X,Y), and pushes them to a multiprocessing `Queue`.
2. **FusionCenter** (in the main process) pulls detections from all cameras, merges points within `--gating` meters, and maintains short-lived tracks (`--max-age` seconds).
3. You get a continuously updated list of IDs and floor coordinates in the console (you can easily extend this to MQTT, WebSocket, etc.).

Prerequisites:
* Calibrate **each** camera (run `quick_start_pipeline.py calibrate` with the correct `--cam`).
* Compute a homography for each camera (`quick_start_pipeline.py homography`) using the *same* floor reference points.
  * Save them as `floor_homography_cam0.npy`, `floor_homography_cam1.npy`, … (the scripts default to that naming scheme).  
  * Likewise for `camera_intrinsics_cam0.npz`, etc.  
  * If only a single set of files (`camera_intrinsics.npz`, `floor_homography.npy`) exists, all workers will fall back to those.

You can adjust fusion parameters at runtime; for example, `--gating 0.5` means detections within 0.5 m are considered the same person. 