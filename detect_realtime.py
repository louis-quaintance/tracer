"""
Real-time golf ball trace visualisation.

Runs YOLO detection on each frame and draws the growing trajectory
line live in a cv2 window. Press 'q' to quit, SPACE to pause/resume.

Usage:
    python detect_realtime.py <video_path> --model best_new_v1.pt [--conf 0.25]
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


def smooth_points(points, window=7):
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


def extend_trajectory(points, num_extra=30):
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


def draw_trajectory(frame, traj, alpha=0.4, thickness=20):
    if len(traj) < 2:
        return frame
    overlay = frame.copy()
    pts = np.array(traj, dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(overlay, [pts], isClosed=False,
                  color=(0, 0, 255), thickness=thickness,
                  lineType=cv2.LINE_AA)
    return cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)


def main():
    parser = argparse.ArgumentParser(description="Real-time golf ball trace")
    parser.add_argument("video", help="Path to input video file")
    parser.add_argument("--model", default="best_new_v1.pt")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--output", "-o", default="realtime_output.mp4",
                        help="Output video file (default: realtime_output.mp4)")
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        print(f"Error: {video_path} not found"); sys.exit(1)

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Error: Model not found at {model_path}"); sys.exit(1)

    print(f"Loading model: {model_path}")
    model = YOLO(str(model_path))

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    delay = max(1, int(1000 / fps))  # ms between frames for real-time playback
    print(f"Video: {w}x{h} @ {fps:.1f} fps, {total} frames")

    # --- Pass 1: detect all frames (no drawing, no display) ---
    print("Pass 1: detecting...")
    frames = []
    trajectory_points = []
    # per-frame index into trajectory_points: how many points exist after this frame
    traj_counts = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        frames.append(frame)

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
                if conf > best_conf:
                    best_conf = conf
                    best_point = (cx, cy)

        if best_point is not None:
            trajectory_points.append(best_point)
        traj_counts.append(len(trajectory_points))

        if frame_idx % 50 == 0:
            print(f"  Frame {frame_idx}/{total} — {len(trajectory_points)} detections")

    cap.release()
    print(f"Detection done: {len(trajectory_points)} detections across {frame_idx} frames")

    # Pre-compute the full smoothed + extended trajectory once
    smooth_traj = smooth_points(trajectory_points)
    full_traj = extend_trajectory(smooth_traj)

    # --- Pass 2: render trajectory onto frames and write video ---
    print("Pass 2: rendering output...")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.output, fourcc, fps, (w, h))

    for i, frame in enumerate(frames):
        # Draw trajectory up to this frame's detection count
        count = traj_counts[i]
        if count >= 2:
            partial = smooth_points(trajectory_points[:count])
            partial = extend_trajectory(partial)
            frame = draw_trajectory(frame, partial)
        writer.write(frame)

    writer.release()
    print(f"Done! Output: {args.output}")


if __name__ == "__main__":
    main()
