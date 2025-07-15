import os
import time
from multiprocessing import Queue
from typing import List, Tuple

import cv2
import numpy as np

from quick_start_pipeline import (
    PersonDetector,
    load_homography,
    load_intrinsics,
)


def camera_worker(
    cam_id: int,
    queue: "Queue",
    homography_file: str | None = None,
    intrinsics_file: str | None = None,
    display: bool = False,
) -> None:
    """Process function that grabs frames, runs detection, projects to floor, and pushes to queue.

    Args:
        cam_id: OpenCV camera index.
        queue: Multiprocessing queue into which a dict will be put every frame:
            {
              'cam_id': int,
              'time':  float (epoch seconds),
              'detections': List[Tuple[float, float, float]]  # (X, Y, conf)
            }
        homography_file: Path to the 3×3 floor homography (defaults to per-cam or global).
        intrinsics_file: Path to `npz` containing camera_matrix & dist_coeffs.
        display: If True, shows the camera feed with overlays.
    """
    # Resolve default file names -------------------------------------------------
    if homography_file is None:
        per_cam = f"floor_homography_cam{cam_id}.npy"
        homography_file = per_cam if os.path.exists(per_cam) else "floor_homography.npy"
    if intrinsics_file is None:
        per_cam = f"camera_intrinsics_cam{cam_id}.npz"
        intrinsics_file = (
            per_cam if os.path.exists(per_cam) else "camera_intrinsics.npz"
        )

    if not (os.path.exists(homography_file) and os.path.exists(intrinsics_file)):
        raise FileNotFoundError(
            "Homography or intrinsics not found for camera worker. Run calibration first."
        )

    camera_matrix, dist_coeffs = load_intrinsics(intrinsics_file)
    H_inv = np.linalg.inv(load_homography(homography_file))

    detector = PersonDetector()

    cap = cv2.VideoCapture(cam_id)
    if not cap.isOpened():
        raise RuntimeError(f"[Cam {cam_id}] Cannot open camera index {cam_id}")

    window_name = f"Camera {cam_id}" if display else None
    if display:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print(f"[Cam {cam_id}] Frame grab failed – stopping worker.")
                break

            frame_u = cv2.undistort(frame, camera_matrix, dist_coeffs)
            detections = detector.detect(frame_u)

            world_points: List[Tuple[float, float, float]] = []
            for xmin, ymin, xmax, ymax, conf in detections:
                u = (xmin + xmax) / 2.0
                v = ymax
                p = np.array([u, v, 1.0])
                P = H_inv @ p
                P /= P[2]
                X, Y = P[:2]
                world_points.append((float(X), float(Y), float(conf)))

                if display:
                    cv2.rectangle(frame_u, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                    cv2.putText(
                        frame_u,
                        f"({X:.2f},{Y:.2f})",
                        (xmin, max(0, ymin - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        1,
                        cv2.LINE_AA,
                    )

            queue.put({
                "cam_id": cam_id,
                "time": time.time(),
                "detections": world_points,
            })

            if display:
                cv2.imshow(window_name, frame_u)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        cap.release()
        if display:
            cv2.destroyWindow(window_name) 