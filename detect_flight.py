"""
Detect golf ball flight in video and overlay the trajectory with flight stats.

Runs YOLO per-frame, links detections into flight segments using simple
nearest-neighbour matching, then draws each flight arc with speed/apex info.

Usage:
    python detect_flight.py <video_path> --model best_v3.pt [--conf 0.3] [--show]
    python detect_flight.py <video_path> --model best_v3.pt -o flight_out.mp4 --slow 3
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


# ── Colours for multiple flights ──
TRAIL_COLOURS = [
    (0, 0, 255),    # red
    (0, 200, 255),  # yellow
    (255, 100, 0),  # blue
    (0, 255, 0),    # green
    (255, 0, 200),  # magenta
]


def dist(a, b):
    return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def detect_balls(model, frame, conf):
    """Return list of (cx, cy, w, h, conf) for all detections in frame."""
    results = model(frame, conf=conf, verbose=False)
    dets = []
    for result in results:
        boxes = result.boxes
        for i in range(len(boxes)):
            bx1, by1, bx2, by2 = map(int, boxes.xyxy[i])
            cx = (bx1 + bx2) // 2
            cy = (by1 + by2) // 2
            dets.append((cx, cy, bx2 - bx1, by2 - by1, float(boxes.conf[i])))
    return dets


class Flight:
    """A single ball flight (sequence of linked detections)."""

    def __init__(self, cx, cy, frame_idx):
        self.points = [(cx, cy, frame_idx)]
        self.lost_frames = 0
        self.finished = False

    @property
    def last_pos(self):
        return self.points[-1][:2]

    @property
    def last_frame(self):
        return self.points[-1][2]

    def add(self, cx, cy, frame_idx):
        self.points.append((cx, cy, frame_idx))
        self.lost_frames = 0

    def mark_lost(self):
        self.lost_frames += 1

    def is_valid_flight(self, min_points=5, min_rise_px=60):
        """True if the ball actually went up (not just sitting on the tee)."""
        if len(self.points) < min_points:
            return False
        ys = [p[1] for p in self.points]
        rise = max(ys) - min(ys)
        return rise >= min_rise_px

    def apex(self):
        """Highest point (smallest y)."""
        return min(self.points, key=lambda p: p[1])

    def total_distance_px(self):
        d = 0
        for i in range(1, len(self.points)):
            d += dist(self.points[i - 1][:2], self.points[i][:2])
        return d

    def max_speed_px(self, fps):
        """Max speed in px/frame, converted to rough px/s."""
        best = 0
        for i in range(1, len(self.points)):
            gap = self.points[i][2] - self.points[i - 1][2]
            if gap == 0:
                continue
            spd = dist(self.points[i - 1][:2], self.points[i][:2]) / gap
            best = max(best, spd)
        return best * fps

    def bounding_rect(self):
        xs = [p[0] for p in self.points]
        ys = [p[1] for p in self.points]
        return min(xs), min(ys), max(xs), max(ys)


def draw_flight(frame, flight, colour, alpha=0.5, thickness=4):
    """Draw a smooth trail for a flight on frame."""
    overlay = frame.copy()
    pts = [(p[0], p[1]) for p in flight.points]
    if len(pts) < 2:
        return

    # Graduated thickness: thin at start, thick at end
    for i in range(1, len(pts)):
        t = max(2, int(thickness * (i / len(pts))))
        cv2.line(overlay, pts[i - 1], pts[i], colour, t, cv2.LINE_AA)

    # Draw small dot at current tip
    cv2.circle(overlay, pts[-1], 6, colour, -1, cv2.LINE_AA)

    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


def draw_flight_stats(frame, flight, colour, fps, idx):
    """Draw stats box next to a flight's apex."""
    apex = flight.apex()
    n_pts = len(flight.points)
    dist_px = flight.total_distance_px()
    max_spd = flight.max_speed_px(fps)
    duration = (flight.points[-1][2] - flight.points[0][2]) / fps

    lines = [
        f"Flight {idx}",
        f"  {n_pts} detections",
        f"  {duration:.1f}s duration",
        f"  {dist_px:.0f}px arc length",
        f"  {max_spd:.0f} px/s peak",
    ]

    # Position stats near the apex
    tx = apex[0] + 20
    ty = max(30, apex[1] - 10)
    for i, line in enumerate(lines):
        y = ty + i * 22
        cv2.putText(frame, line, (tx, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, line, (tx, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 1, cv2.LINE_AA)


def main():
    parser = argparse.ArgumentParser(description="Detect golf ball flight in video")
    parser.add_argument("video", help="Path to input video file")
    parser.add_argument("--model", default="best_v3.pt")
    parser.add_argument("--output", "-o", default="flight_output.mp4")
    parser.add_argument("--conf", type=float, default=0.3,
                        help="YOLO confidence threshold")
    parser.add_argument("--max-gap", type=int, default=12,
                        help="Max frames without detection before ending a flight")
    parser.add_argument("--link-dist", type=int, default=150,
                        help="Max pixel distance to link a detection to an active flight")
    parser.add_argument("--slow", type=int, default=1,
                        help="Slow-motion factor for output (e.g. 3 = 3x slower)")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--debug", action="store_true",
                        help="Show per-frame detections and linking info")
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        print(f"Error: {video_path} not found")
        sys.exit(1)

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        sys.exit(1)

    print(f"Loading model: {model_path}")
    model = YOLO(str(model_path))

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {w}x{h} @ {fps:.1f} fps, {total} frames")

    # ── Pass 1: detect all balls and build flights ──
    print("Pass 1: Detecting golf balls...")
    active_flights = []
    finished_flights = []

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        dets = detect_balls(model, frame, args.conf)

        # Try to link each detection to the nearest active flight
        used = set()
        for flight in active_flights:
            if flight.finished:
                continue
            best_d = args.link_dist
            best_det = None
            for j, det in enumerate(dets):
                if j in used:
                    continue
                d = dist(flight.last_pos, det[:2])
                if d < best_d:
                    best_d = d
                    best_det = j
            if best_det is not None:
                det = dets[best_det]
                flight.add(det[0], det[1], frame_idx)
                used.add(best_det)
            else:
                flight.mark_lost()

        # End flights that have been lost too long
        still_active = []
        for flight in active_flights:
            if flight.lost_frames >= args.max_gap:
                flight.finished = True
                if flight.is_valid_flight():
                    finished_flights.append(flight)
            else:
                still_active.append(flight)
        active_flights = still_active

        # Start new flights for unlinked detections
        for j, det in enumerate(dets):
            if j not in used:
                active_flights.append(Flight(det[0], det[1], frame_idx))

        if frame_idx % 100 == 0:
            print(f"  Frame {frame_idx}/{total} — "
                  f"{len(active_flights)} active, {len(finished_flights)} complete")

    # Finish remaining active flights
    for flight in active_flights:
        if flight.is_valid_flight():
            finished_flights.append(flight)

    cap.release()

    # Sort flights by length (most points first)
    finished_flights.sort(key=lambda f: len(f.points), reverse=True)
    print(f"\nFound {len(finished_flights)} valid flight(s)")
    for i, f in enumerate(finished_flights):
        dur = (f.points[-1][2] - f.points[0][2]) / fps
        print(f"  Flight {i+1}: {len(f.points)} pts, {dur:.1f}s, "
              f"frames {f.points[0][2]}–{f.points[-1][2]}")

    if not finished_flights:
        print("No flights detected. Try lowering --conf or check your model.")
        sys.exit(0)

    # ── Pass 2: write output video with trails ──
    print("\nPass 2: Writing output video...")
    cap2 = cv2.VideoCapture(str(video_path))
    out_fps = fps / args.slow
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(args.output, fourcc, out_fps, (w, h))

    frame_idx = 0
    while True:
        ret, frame = cap2.read()
        if not ret:
            break
        frame_idx += 1

        # Draw each flight's trail up to the current frame
        for i, flight in enumerate(finished_flights):
            colour = TRAIL_COLOURS[i % len(TRAIL_COLOURS)]

            # Only draw if we're within the flight's time window
            start_f = flight.points[0][2]
            end_f = flight.points[-1][2]
            if frame_idx < start_f:
                continue

            # Build partial flight up to current frame
            visible_pts = [p for p in flight.points if p[2] <= frame_idx]
            if len(visible_pts) < 2:
                continue

            partial = Flight(visible_pts[0][0], visible_pts[0][1], visible_pts[0][2])
            partial.points = visible_pts

            draw_flight(frame, partial, colour, alpha=0.5, thickness=6)

            # Show stats once flight is complete
            if frame_idx >= end_f:
                draw_flight_stats(frame, flight, colour, fps, i + 1)

        out.write(frame)

        if args.show:
            cv2.imshow("Golf Ball Flight", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap2.release()
    out.release()
    cv2.destroyAllWindows()

    print(f"\nDone! Output: {args.output}")
    if args.slow > 1:
        print(f"  Slow motion: {args.slow}x ({out_fps:.1f} fps)")
    print(f"  Flights detected: {len(finished_flights)}")


if __name__ == "__main__":
    main()
