import argparse
import math
import os
import time
from multiprocessing import Process, Queue
from typing import Dict, List

from detect_cam import camera_worker


# ------------------------------------------------------------
# Simple world-space tracker (distance gating)
# ------------------------------------------------------------
class Track:
    __slots__ = ("track_id", "x", "y", "last_seen")

    def __init__(self, track_id: int, x: float, y: float):
        self.track_id = track_id
        self.x = x
        self.y = y
        self.last_seen = time.time()

    def update(self, x: float, y: float) -> None:
        self.x = x
        self.y = y
        self.last_seen = time.time()


class FusionCenter:
    """Fuses detections from multiple cameras into world-space tracks."""

    def __init__(self, gating: float = 0.75, max_age: float = 1.0):
        self.gating = gating  # meters
        self.max_age = max_age
        self.tracks: List[Track] = []
        self._next_id = 0

    def _match_track(self, x: float, y: float) -> Track | None:
        for trk in self.tracks:
            if math.hypot(trk.x - x, trk.y - y) < self.gating:
                return trk
        return None

    def ingest(self, detections: List[tuple[float, float, float]]):
        """Update tracks with a batch of (X, Y, conf)."""
        for x, y, _conf in detections:
            trk = self._match_track(x, y)
            if trk:
                trk.update(x, y)
            else:
                self.tracks.append(Track(self._next_id, x, y))
                self._next_id += 1

        # prune stale tracks
        now = time.time()
        self.tracks = [t for t in self.tracks if now - t.last_seen < self.max_age]

    def summary_str(self) -> str:
        return "\n".join(
            f"ID {t.track_id}: ({t.x:.2f}, {t.y:.2f}) m" for t in self.tracks
        )


# ------------------------------------------------------------
# Orchestration
# ------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run multi-camera person detection & fusion into a common floor plane"
    )
    parser.add_argument(
        "--cams",
        type=int,
        nargs="+",
        default=[0, 1],
        help="List of camera indices to launch (e.g. --cams 0 1 2)",
    )
    parser.add_argument(
        "--display",
        action="store_true",
        help="Show each camera's window (press q in any to quit)",
    )
    parser.add_argument(
        "--gating",
        type=float,
        default=0.75,
        help="Distance (m) threshold to consider two detections the same person",
    )
    parser.add_argument(
        "--max-age",
        type=float,
        default=1.0,
        help="Seconds after which a lost track is removed",
    )

    args = parser.parse_args()

    # Queue for inter-process communication
    q: Queue = Queue()

    # Spawn one worker per camera
    processes: List[Process] = []
    for cam_id in args.cams:
        p = Process(
            target=camera_worker,
            args=(cam_id, q),
            kwargs={"display": args.display},
            daemon=True,
        )
        p.start()
        processes.append(p)
        print(f"[Main] Spawned camera worker for cam {cam_id} (pid {p.pid})")

    fusion = FusionCenter(gating=args.gating, max_age=args.max_age)

    try:
        while True:
            try:
                msg: Dict = q.get(timeout=0.1)  # blocks briefly
            except Exception:
                # no message, just refresh display
                msg = None

            if msg is not None:
                fusion.ingest(msg["detections"])

            # Clear console & print current tracks every ~100 ms
            os.system("clear")
            print("Active tracks (Ctrl-C to stop):\n")
            print(fusion.summary_str() or "<none>")
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n[Main] Stopping …")
    finally:
        for p in processes:
            p.terminate()
            p.join()
        q.close()


if __name__ == "__main__":
    main() 