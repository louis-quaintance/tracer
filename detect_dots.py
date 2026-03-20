"""
Draw red dots wherever the model detects a golf ball in every frame,
and connect detected centres with a smooth red line.

Usage:
    python detect_dots.py <video_path> --model best_v2.pt [--conf 0.25] [--show]
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

def add_end_loop(points, loop_radius=10, num_loop_points=12):
    """
    Add a small curved hook/loop to the end of the trajectory
    to mimic a stylised golf-shot tracer finish.
    """
    if len(points) < 2:
        return points

    p_last = np.array(points[-1], dtype=np.float32)
    p_prev = np.array(points[-2], dtype=np.float32)

    direction = p_last - p_prev
    norm = np.linalg.norm(direction)
    if norm < 1e-6:
        return points

    direction = direction / norm

    # Perpendicular vector
    perp = np.array([-direction[1], direction[0]], dtype=np.float32)

    # Make a small downward curling arc
    loop_points = []
    for t in np.linspace(0, 1, num_loop_points):
        angle = t * np.pi * 0.9  # partial arc, not full circle

        forward = direction * (t * loop_radius * 0.8)
        sideways = perp * (np.sin(angle) * loop_radius * 0.6)
        downward = np.array([0, (t ** 1.5) * loop_radius * 1.2], dtype=np.float32)

        pt = p_last + forward + sideways + downward
        loop_points.append((int(pt[0]), int(pt[1])))

    return points + loop_points

def extend_trajectory(points, num_extra=10):
    """
    Extend trajectory forward using last known velocity.
    """
    if len(points) < 2:
        return points

    p1 = np.array(points[-2])
    p2 = np.array(points[-1])

    velocity = p2 - p1

    extended = points.copy()
    current = p2.copy()

    for _ in range(num_extra):
        current = current + velocity
        extended.append((int(current[0]), int(current[1])))

    return extended

def smooth_points(points, window=5):
    """
    Apply a simple moving average to point coordinates.
    Keeps the same number of points.
    """
    if len(points) < 3:
        return points

    smoothed = []
    half = window // 2

    for i in range(len(points)):
        start = max(0, i - half)
        end = min(len(points), i + half + 1)
        chunk = points[start:end]

        avg_x = int(sum(p[0] for p in chunk) / len(chunk))
        avg_y = int(sum(p[1] for p in chunk) / len(chunk))
        smoothed.append((avg_x, avg_y))

    return smoothed


def main():
    parser = argparse.ArgumentParser(description="Draw red dots on detected golf balls")
    parser.add_argument("video", help="Path to input video file")
    parser.add_argument("--model", default="best_new_v1.pt")
    parser.add_argument("--output", "-o", default="dots_output.mp4")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--show", action="store_true")
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

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(args.output, fourcc, fps, (w, h))

    frame_idx = 0
    detections_total = 0

    # Keep all detected centres across frames
    trajectory_points = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        results = model(frame, conf=args.conf, verbose=False)

        best_point = None
        best_conf = -1.0

        for result in results:
            boxes = result.boxes
            for i in range(len(boxes)):
                bx1, by1, bx2, by2 = map(int, boxes.xyxy[i])
                cx = (bx1 + bx2) // 2
                cy = (by1 + by2) // 2
                conf = float(boxes.conf[i])

                # Pick the highest-confidence ball in this frame
                if conf > best_conf:
                    best_conf = conf
                    best_point = (cx, cy)

        # Add only one point per frame
        if best_point is not None:
            trajectory_points.append(best_point)
            detections_total += 1

        # Smooth the trajectory
        # Smooth trajectory
        smooth_traj = smooth_points(trajectory_points, window=7)

        # Extend forward
        extended_traj = extend_trajectory(smooth_traj, num_extra=12)

        final_traj = add_end_loop(extended_traj, loop_radius=20, num_loop_points=30)

        # Create overlay for transparency
        overlay = frame.copy()

        if len(final_traj) >= 2:
            pts = np.array(final_traj, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(
                overlay,
                [pts],
                isClosed=False,
                color=(0, 0, 255),
                thickness=8,
                lineType=cv2.LINE_AA,
            )

        # Blend overlay with original frame (transparency)
        alpha = 0.4  # 0 = invisible, 1 = solid
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

        out.write(frame)

        if args.show:
            cv2.imshow("Detections", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        if frame_idx % 50 == 0:
            print(f"  Frame {frame_idx}/{total} — {detections_total} detections so far")

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"\nDone! {detections_total} total detections across {frame_idx} frames")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()