"""
Golf ball tracker — YOLOv8 detection + Kalman filter tracking.

What this version improves:
- Uses last REAL detection for gating, not coasted Kalman points
- Adds box geometry filtering (area / aspect ratio)
- Uses distance gating around prediction
- Uses frame-skip-aware Kalman dt
- Smooths final trail before drawing
- Filters jumpy trail points
- Makes search zone configurable
- Uses more realistic Kalman measurement noise for tiny object detection

Usage:
    python golf_tracer.py <video_path> --model best.pt [--show] [--debug]

Example:
    python golf_tracer.py input.mp4 --model best.pt --output traced.mp4 --show --debug
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


class BallKalmanFilter:
    def __init__(self, x, y, dt=1.0):
        self.dt = float(dt)
        self.xhat = np.array([x, y, 0.0, 0.0], dtype=float)

        # State covariance
        self.P = np.eye(4) * 50.0

        # State transition
        self.F = np.array([
            [1, 0, self.dt, 0],
            [0, 1, 0, self.dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=float)

        # Measurement model
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=float)

        # Process noise: allow velocity to change more than position
        self.Q = np.diag([2.0, 2.0, 8.0, 8.0])

        # Measurement noise: YOLO box center on a tiny blurred ball is noisy
        self.R = np.diag([3.0, 3.0])

        self.age = 0
        self.missed = 0

    def predict(self):
        self.xhat = self.F @ self.xhat
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.age += 1
        return self.xhat[:2].copy()

    def update(self, z):
        z = np.array(z, dtype=float)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.xhat = self.xhat + K @ (z - self.H @ self.xhat)
        self.P = (np.eye(4) - K @ self.H) @ self.P
        self.missed = 0

    def mark_missed(self):
        self.missed += 1

    @property
    def position(self):
        return int(round(self.xhat[0])), int(round(self.xhat[1]))

    @property
    def speed(self):
        return float(np.hypot(self.xhat[2], self.xhat[3]))


def smooth_trail(points, window=5):
    """
    Smooth list of (x, y, frame_idx) using moving average on x/y only.
    """
    if len(points) < 3 or window < 2:
        return points

    arr = np.array(points, dtype=float)
    smoothed = []
    half = window // 2

    for i in range(len(arr)):
        a = max(0, i - half)
        b = min(len(arr), i + half + 1)
        mean_xy = arr[a:b, :2].mean(axis=0)
        smoothed.append((int(round(mean_xy[0])),
                         int(round(mean_xy[1])),
                         int(arr[i, 2])))
    return smoothed


def filter_trail_points(points, max_step=60):
    """
    Remove jumpy detections from a trail.
    Input: [(x, y, frame_idx), ...]
    """
    if len(points) < 2:
        return points

    filtered = [points[0]]
    for p in points[1:]:
        x1, y1, _ = filtered[-1]
        x2, y2, _ = p
        if np.hypot(x2 - x1, y2 - y1) <= max_step:
            filtered.append(p)
    return filtered


def is_upward_trajectory(real_detections, launch_y, min_real=4, min_rise=80,
                         upward_ratio=0.65):
    """
    Only counts real YOLO detections (not Kalman coasted positions).
    Requires:
      - At least min_real actual detections
      - Ball moved upward by at least min_rise px
      - Detections are spread out vertically
      - Majority of steps move upward
    """
    if len(real_detections) < min_real or launch_y is None:
        return False

    ys = [p[1] for p in real_detections]
    rise = launch_y - min(ys)
    if rise < min_rise:
        return False

    y_spread = max(ys) - min(ys)
    if y_spread < min_rise * 0.5:
        return False

    upward_steps = 0
    total_steps = 0
    for i in range(1, len(real_detections)):
        dy = real_detections[i][1] - real_detections[i - 1][1]
        total_steps += 1
        if dy < 0:
            upward_steps += 1

    if total_steps == 0:
        return False

    return (upward_steps / total_steps) >= upward_ratio


def select_best_detection(results, offset_x, offset_y, pred_x=None, pred_y=None,
                          last_real_pos=None, launch_pos=None,
                          full_w=None,
                          min_area=4, max_area=400,
                          min_ar=0.5, max_ar=1.8,
                          max_dist=None,
                          allow_downward_px=20,
                          max_launch_dx_ratio=0.35):
    """
    Pick best detection from a YOLO result set using ball-specific geometry and motion gating.
    Returns: (cx, cy, bw, bh, conf) or None
    """
    best = None
    best_score = -1e9

    for result in results:
        boxes = result.boxes
        for i in range(len(boxes)):
            c = float(boxes.conf[i])
            bx1, by1, bx2, by2 = map(int, boxes.xyxy[i])

            bw = bx2 - bx1
            bh = by2 - by1
            if bw <= 0 or bh <= 0:
                continue

            area = bw * bh
            if area < min_area or area > max_area:
                continue

            ar = bw / bh
            if ar < min_ar or ar > max_ar:
                continue

            cx = (bx1 + bx2) // 2 + offset_x
            cy = (by1 + by2) // 2 + offset_y

            # Gate by distance to prediction
            dist = 0.0
            if pred_x is not None and pred_y is not None:
                dist = float(np.hypot(cx - pred_x, cy - pred_y))
                if max_dist is not None and dist > max_dist:
                    continue

            # Do not allow large downward moves once tracking
            if last_real_pos is not None:
                if cy > last_real_pos[1] + allow_downward_px:
                    continue

            # Ball should not suddenly jump far sideways from launch line
            if launch_pos is not None and full_w is not None:
                if abs(cx - launch_pos[0]) > full_w * max_launch_dx_ratio:
                    continue

            # Score = confidence + closeness + slight preference for square-ish boxes
            score = 2.0 * c
            score -= 0.015 * dist
            score -= 0.002 * abs(bw - bh)

            if last_real_pos is not None and cy < last_real_pos[1]:
                score += 0.15  # slight upward preference

            if score > best_score:
                best_score = score
                best = (cx, cy, bw, bh, c)

    return best


def detect_in_region(model, frame, cx, cy, crop_size, conf, full_w, full_h,
                     pred_x=None, pred_y=None,
                     last_real_pos=None, launch_pos=None):
    """
    Crop around predicted point, run YOLO, return best detection.
    """
    half = crop_size // 2

    if last_real_pos is not None:
        # Bias crop upward while tracking
        x1 = max(0, cx - half)
        y1 = max(0, cy - int(half * 1.2))
        x2 = min(full_w, cx + half)
        y2 = min(full_h, cy + int(half * 0.5))
    else:
        x1 = max(0, cx - half)
        y1 = max(0, cy - half)
        x2 = min(full_w, cx + half)
        y2 = min(full_h, cy + half)

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None, (x1, y1, x2, y2)

    results = model(crop, conf=conf, verbose=False)

    max_dist = crop_size * 0.6
    best = select_best_detection(
        results,
        offset_x=x1,
        offset_y=y1,
        pred_x=pred_x,
        pred_y=pred_y,
        last_real_pos=last_real_pos,
        launch_pos=launch_pos,
        full_w=full_w,
        min_area=4,
        max_area=400,
        min_ar=0.5,
        max_ar=1.8,
        max_dist=max_dist,
        allow_downward_px=20,
        max_launch_dx_ratio=0.35,
    )

    return best, (x1, y1, x2, y2)


def main():
    parser = argparse.ArgumentParser(description="Golf ball trajectory tracer")
    parser.add_argument("video", help="Path to input video file")
    parser.add_argument("--model", default="best.pt")
    parser.add_argument("--output", "-o", default="traced_output.mp4")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--crop-size", type=int, default=200)
    parser.add_argument("--max-lost", type=int, default=15)
    parser.add_argument("--target-fps", type=int, default=30,
                        help="Max FPS to process (skips frames if input is higher)")
    parser.add_argument("--all-trails", action="store_true",
                        help="Draw all detected trails, not just the best")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--debug", action="store_true")

    # Search zone config
    parser.add_argument("--search-x-min", type=float, default=0.35)
    parser.add_argument("--search-x-max", type=float, default=0.65)
    parser.add_argument("--search-y-min", type=float, default=0.50)
    parser.add_argument("--search-y-max", type=float, default=0.95)

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

    # Frame skipping
    frame_skip = max(1, int(round(fps / args.target_fps))) if fps > args.target_fps else 1
    out_fps = fps / frame_skip if frame_skip > 0 else fps
    if frame_skip > 1:
        print(f"Input {fps:.0f}fps > target {args.target_fps}fps — "
              f"processing every {frame_skip} frame(s), output at {out_fps:.1f}fps")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    trail_thickness = max(4, int(min(w, h) * 0.008))

    # Search zone
    search_x1 = int(w * args.search_x_min)
    search_x2 = int(w * args.search_x_max)
    search_y1 = int(h * args.search_y_min)
    search_y2 = int(h * args.search_y_max)
    print(f"Search zone: ({search_x1},{search_y1}) to ({search_x2},{search_y2})")

    INIT_CONF = max(args.conf, 0.5)

    # Tracking state
    kf = None
    all_trails = []      # list of dicts: {start, end, raw, smooth}
    current_trail = []   # debug only: includes coasted points
    real_detections = [] # list of (x, y, frame_idx)

    launch_y = None
    launch_pos = None
    last_real_pos = None

    moving_up = False
    moving_up_frame = None

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        if frame_skip > 1 and (frame_idx % frame_skip) != 0:
            continue

        detection = None
        crop_rect = None

        if kf is None:
            # Initial search only in tee area
            search_crop = frame[search_y1:search_y2, search_x1:search_x2]
            results = model(search_crop, conf=INIT_CONF, verbose=False)

            detection = select_best_detection(
                results,
                offset_x=search_x1,
                offset_y=search_y1,
                pred_x=None,
                pred_y=None,
                last_real_pos=None,
                launch_pos=None,
                full_w=w,
                min_area=4,
                max_area=500,
                min_ar=0.5,
                max_ar=1.8,
                max_dist=None,
            )

            if detection is not None:
                cx, cy = detection[0], detection[1]
                kf = BallKalmanFilter(cx, cy, dt=frame_skip)
                current_trail = [(cx, cy)]
                real_detections = [(cx, cy, frame_idx)]
                launch_y = cy
                launch_pos = (cx, cy)
                last_real_pos = (cx, cy)

                print(f"  Frame {frame_idx}: Ball found at ({cx},{cy}) conf={detection[4]:.2f}")

        else:
            predicted = kf.predict()
            pred_x, pred_y = int(round(predicted[0])), int(round(predicted[1]))

            # Dynamic crop
            speed = kf.speed
            crop = args.crop_size
            if speed > 30:
                crop = int(crop * 1.5)
            if kf.missed > 3:
                crop = int(crop * (1 + kf.missed * 0.2))
            crop = min(crop, max(w, h))

            # Lower conf during tracking, but clamp to safer range
            track_conf = max(0.08, min(args.conf * 0.4, 0.18))

            detection, crop_rect = detect_in_region(
                model=model,
                frame=frame,
                cx=pred_x,
                cy=pred_y,
                crop_size=crop,
                conf=track_conf,
                full_w=w,
                full_h=h,
                pred_x=pred_x,
                pred_y=pred_y,
                last_real_pos=last_real_pos,
                launch_pos=launch_pos,
            )

            if detection is not None:
                cx, cy = detection[0], detection[1]
                kf.update([cx, cy])

                current_trail.append((cx, cy))
                real_detections.append((cx, cy, frame_idx))
                last_real_pos = (cx, cy)

                if not moving_up and launch_y is not None and cy < launch_y - 20:
                    moving_up = True
                    moving_up_frame = frame_idx
            else:
                kf.mark_missed()
                px, py = kf.position
                if 0 <= px < w and 0 <= py < h:
                    current_trail.append((px, py))

            # End current candidate trail
            if kf.missed >= args.max_lost or (moving_up and kf.missed >= 10):
                if is_upward_trajectory(real_detections, launch_y):
                    filtered = filter_trail_points(real_detections, max_step=60)
                    smoothed = smooth_trail(filtered, window=5)

                    start_f = moving_up_frame if moving_up_frame is not None else (
                        real_detections[0][2] if real_detections else frame_idx
                    )

                    all_trails.append({
                        "start": start_f,
                        "end": frame_idx,
                        "raw": list(real_detections),
                        "smooth": smoothed,
                    })
                    print(f"  Frame {frame_idx}: Shot captured! "
                          f"{len(real_detections)} real detections")
                else:
                    print(f"  Frame {frame_idx}: Discarded trail "
                          f"({len(current_trail)} pts, {len(real_detections)} real) — not upward")

                # Reset tracker
                kf = None
                current_trail = []
                real_detections = []
                launch_y = None
                launch_pos = None
                last_real_pos = None
                moving_up = False
                moving_up_frame = None

        # Debug drawing
        if args.debug:
            if kf is None:
                cv2.rectangle(frame, (search_x1, search_y1),
                              (search_x2, search_y2), (255, 0, 0), 2)
                cv2.putText(frame, "SEARCH ZONE", (search_x1, search_y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

            if crop_rect is not None:
                rx1, ry1, rx2, ry2 = crop_rect
                cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (255, 100, 0), 1)

            if kf is not None:
                px, py = kf.position
                cv2.drawMarker(frame, (px, py), (0, 100, 255),
                               cv2.MARKER_CROSS, 15, 1)

                if last_real_pos is not None:
                    cv2.circle(frame, last_real_pos, 4, (0, 255, 0), -1)

        if frame_idx % 50 == 0:
            state = "tracking" if kf else "searching"
            print(f"  Frame {frame_idx}/{total} [{state}] "
                  f"trail={len(current_trail)} shots={len(all_trails)}")

    # Save final trail if video ends mid-track
    if real_detections and is_upward_trajectory(real_detections, launch_y):
        filtered = filter_trail_points(real_detections, max_step=60)
        smoothed = smooth_trail(filtered, window=5)
        start_f = moving_up_frame if moving_up_frame is not None else real_detections[0][2]
        all_trails.append({
            "start": start_f,
            "end": frame_idx,
            "raw": list(real_detections),
            "smooth": smoothed,
        })

    cap.release()

    # Select which trails to draw
    if all_trails:
        if args.all_trails:
            draw_trails = all_trails
            print(f"\nDrawing all {len(all_trails)} trail(s)")
        else:
            draw_trails = [max(all_trails, key=lambda t: len(t["smooth"]))]
            print(f"\nBest shot: {len(draw_trails[0]['smooth'])} points "
                  f"(frames {draw_trails[0]['start']}–{draw_trails[0]['end']}, "
                  f"from {len(all_trails)} candidate(s))")

        for i, t in enumerate(all_trails):
            print(f"  Trail {i+1}: {len(t['smooth'])} points "
                  f"(frames {t['start']}–{t['end']})")
    else:
        draw_trails = []
        print("\nNo valid upward trajectories found.")

    # Second pass: write output
    print("Writing output video...")
    cap2 = cv2.VideoCapture(str(video_path))
    out = cv2.VideoWriter(args.output, fourcc, out_fps, (w, h))

    frame_idx = 0
    while True:
        ret, frame = cap2.read()
        if not ret:
            break
        frame_idx += 1

        if frame_skip > 1 and (frame_idx % frame_skip) != 0:
            continue

        overlay = frame.copy()
        drew = False

        for trail_info in draw_trails:
            trail = trail_info["smooth"]
            pts = [(x, y) for x, y, f in trail if f <= frame_idx]

            if len(pts) > 1:
                for i in range(1, len(pts)):
                    cv2.line(overlay, pts[i - 1], pts[i],
                             (0, 0, 255), trail_thickness, cv2.LINE_AA)
                drew = True

        if drew:
            cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

        out.write(frame)

        if args.show:
            cv2.imshow("Golf Ball Tracker", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap2.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Done! Output: {args.output}")


if __name__ == "__main__":
    main()