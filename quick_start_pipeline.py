import argparse
import os
import sys
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

# Optional heavy imports guarded to allow steps like calibration to run without them
try:
    import torch  # noqa: E402
except ImportError:
    torch = None  # type: ignore

try:
    from sort import Sort  # noqa: E402
except ImportError:
    Sort = None  # type: ignore

CHECKERBOARD = (9, 6)  # inner corners per a chessboard row and column
SQUARE_SIZE = 0.025  # meters per square – adjust to your printed board
CALIBRATION_FILE = "camera_intrinsics.npz"
HOMOGRAPHY_FILE = "floor_homography.npy"

######################################################################
# Utilities
######################################################################

def draw_text(img: np.ndarray, text: str, org=(10, 30), color=(0, 255, 0)) -> None:
    """Helper to overlay text on an image."""
    cv2.putText(
        img,
        text,
        org,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2,
        cv2.LINE_AA,
    )


def save_intrinsics(camera_matrix: np.ndarray, dist_coeffs: np.ndarray, path: str = CALIBRATION_FILE) -> None:
    np.savez(path, camera_matrix=camera_matrix, dist_coeffs=dist_coeffs)
    print(f"[INFO] Saved intrinsics to {path}")


def load_intrinsics(path: str = CALIBRATION_FILE) -> Tuple[np.ndarray, np.ndarray]:
    data = np.load(path)
    return data["camera_matrix"], data["dist_coeffs"]


def save_homography(H: np.ndarray, path: str = HOMOGRAPHY_FILE) -> None:
    np.save(path, H)
    print(f"[INFO] Saved homography to {path}")


def load_homography(path: str = HOMOGRAPHY_FILE) -> np.ndarray:
    return np.load(path)


######################################################################
# Camera Calibration
######################################################################

def calibrate_camera(cam_id: int = 0, frames_to_collect: int = 20) -> None:
    """Collects checkerboard images and calibrates the camera."""
    cap = cv2.VideoCapture(cam_id)
    if not cap.isOpened():
        print("[ERROR] Cannot open camera")
        return

    objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0 : CHECKERBOARD[0], 0 : CHECKERBOARD[1]].T.reshape(-1, 2)
    objp *= SQUARE_SIZE

    objpoints: List[np.ndarray] = []  # 3d point in real world space
    imgpoints: List[np.ndarray] = []  # 2d points in image plane.

    print("[INFO] Press 'c' to capture frame when checkerboard is visible. 'q' to quit.")
    collected = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret_corners, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

        if ret_corners:
            cv2.drawChessboardCorners(frame, CHECKERBOARD, corners, ret_corners)

        draw_text(frame, f"Collected {collected}/{frames_to_collect}")
        cv2.imshow("Calibration", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("c") and ret_corners:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria=(
                cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                30,
                0.001,
            ))
            imgpoints.append(corners2)
            collected += 1
            print(f"[INFO] Frame captured ({collected}/{frames_to_collect})")
        elif key == ord("q"):
            break
        if collected >= frames_to_collect:
            print("[INFO] Collected required frames.")
            break

    cap.release()
    cv2.destroyAllWindows()

    if len(objpoints) < 5:
        print("[ERROR] Not enough frames for calibration.")
        return

    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None
    )
    print(f"[INFO] Calibration RMS error: {ret}")

    save_intrinsics(camera_matrix, dist_coeffs)


######################################################################
# Homography Computation
######################################################################

def compute_homography(cam_id: int = 0) -> None:
    """Interactively select 4 floor points in the image and enter their real-world coords."""
    if not os.path.exists(CALIBRATION_FILE):
        print("[ERROR] Intrinsics not found. Run calibration first.")
        return

    camera_matrix, dist_coeffs = load_intrinsics()

    cap = cv2.VideoCapture(cam_id)
    if not cap.isOpened():
        print("[ERROR] Cannot open camera")
        return

    ret, frame = cap.read()
    cap.release()
    if not ret:
        print("[ERROR] Failed to grab frame")
        return

    h, w = frame.shape[:2]
    undistorted = cv2.undistort(frame, camera_matrix, dist_coeffs)

    points_img: List[Tuple[int, int]] = []

    def click_event(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points_img.append((x, y))
            cv2.circle(undistorted, (x, y), 5, (0, 0, 255), -1)
            cv2.imshow("Select 4 floor points (clockwise)", undistorted)
            print(f"[INFO] Selected image point: {(x, y)}")

    cv2.imshow("Select 4 floor points (clockwise)", undistorted)
    cv2.setMouseCallback("Select 4 floor points (clockwise)", click_event)

    while len(points_img) < 4:
        cv2.waitKey(1)
    cv2.destroyAllWindows()

    print("[INFO] Enter corresponding real-world X Y coordinates (meters):")
    points_world: List[Tuple[float, float]] = []
    for i in range(4):
        xw, yw = map(float, input(f"World point {i+1} (format: X Y): ").split())
        points_world.append((xw, yw))

    pts_img_np = np.array(points_img, dtype=np.float32)
    pts_world_np = np.array(points_world, dtype=np.float32)

    # Homography maps world -> image; we need inverse later
    H, status = cv2.findHomography(pts_world_np, pts_img_np)
    save_homography(H)

######################################################################
# Person Detector
######################################################################

class PersonDetector:
    def __init__(self, device: str = "cpu") -> None:
        if torch is None:
            raise ImportError("PyTorch not installed. Install torch to use YOLOv5 detector.")
        self.device = device
        print("[INFO] Loading YOLOv5n model, this may take a while the first time …")
        self.model = torch.hub.load("ultralytics/yolov5", "yolov5n", pretrained=True)
        self.model.to(device)
        self.model.eval()

    def detect(self, frame: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
        """Returns list of (xmin, ymin, xmax, ymax, conf) for person class only."""
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.model(img_rgb, size=640)
        detections = []
        for *box, conf, cls in results.xyxy[0].cpu().numpy():
            if int(cls) == 0:  # class 0 is person in COCO
                xmin, ymin, xmax, ymax = map(int, box)
                detections.append((xmin, ymin, xmax, ymax, float(conf)))
        return detections

######################################################################
# Main Loop – Detection + Tracking + Projection
######################################################################

def run_detection(cam_id: int = 0, track: bool = True) -> None:
    if not (os.path.exists(CALIBRATION_FILE) and os.path.exists(HOMOGRAPHY_FILE)):
        print("[ERROR] Calibration or homography files not found. Run those steps first.")
        return

    camera_matrix, dist_coeffs = load_intrinsics()
    H = load_homography()
    H_inv = np.linalg.inv(H)

    detector = PersonDetector()
    tracker = Sort() if track and Sort is not None else None

    cap = cv2.VideoCapture(cam_id)
    if not cap.isOpened():
        print("[ERROR] Cannot open camera")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        undistorted = cv2.undistort(frame, camera_matrix, dist_coeffs)

        detections = detector.detect(undistorted)
        det_array = np.array([[d[0], d[1], d[2], d[3], d[4]] for d in detections]) if detections else np.empty((0, 5))

        tracks = tracker.update(det_array) if tracker else det_array
        for trk in tracks:
            xmin, ymin, xmax, ymax, track_id = map(int, trk[:5])
            u = (xmin + xmax) / 2.0
            v = ymax
            pixel_p = np.array([u, v, 1.0])
            world_p = H_inv @ pixel_p
            world_p /= world_p[2]
            X, Y = world_p[:2]

            # Draw on image
            cv2.rectangle(undistorted, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            draw_text(
                undistorted,
                f"ID {track_id} @ ({X:.2f}, {Y:.2f}) m",
                org=(xmin, ymin - 10),
            )

        cv2.imshow("Detection", undistorted)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

######################################################################
# CLI
######################################################################

def main() -> None:
    parser = argparse.ArgumentParser(description="Quick-Start Person Detection & Tracking Pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    calib_parser = subparsers.add_parser("calibrate", help="Calibrate camera intrinsics with a checkerboard")
    calib_parser.add_argument("--cam", type=int, default=0, help="Camera index (default 0)")
    calib_parser.add_argument("--frames", type=int, default=20, help="Frames to collect for calibration")

    hom_parser = subparsers.add_parser("homography", help="Compute floor homography interactively")
    hom_parser.add_argument("--cam", type=int, default=0, help="Camera index (default 0)")

    detect_parser = subparsers.add_parser("detect", help="Run person detection & tracking")
    detect_parser.add_argument("--cam", type=int, default=0, help="Camera index (default 0)")
    detect_parser.add_argument("--no-track", action="store_true", help="Disable tracking/SORT")

    args = parser.parse_args()

    if args.command == "calibrate":
        calibrate_camera(args.cam, args.frames)
    elif args.command == "homography":
        compute_homography(args.cam)
    elif args.command == "detect":
        run_detection(args.cam, track=not args.no_track)
    else:
        parser.print_help()


if __name__ == "__main__":
    main() 