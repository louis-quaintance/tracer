"""
Golf ball tracker — YOLOv8 detection + Kalman filter tracking.

Initial search is restricted to the bottom-centre of the frame
(central 30% width, bottom 50% height). Once found, tracks upward only.

Usage:
    python golf_tracer.py <video_path> --model best.pt [--show] [--debug]
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


class BallKalmanFilter:
    def __init__(self, x, y):
        self.xhat = np.array([x, y, 0.0, 0.0], dtype=float)
        self.P = np.eye(4) * 100.0
        self.F = np.array([
            [1, 0, 1, 0], [0, 1, 0, 1],
            [0, 0, 1, 0], [0, 0, 0, 1],
        ], dtype=float)
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=float)
        self.Q = np.eye(4) * 5.0
        self.R = np.eye(2) * 0.5
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
        return int(self.xhat[0]), int(self.xhat[1])

    @property
    def speed(self):
        return np.sqrt(self.xhat[2] ** 2 + self.xhat[3] ** 2)


def detect_in_region(model, frame, cx, cy, crop_size, conf, full_w, full_h,
                     last_pos=None, launch_pos=None):
    """
    Crop around (cx, cy), run YOLO, return best detection.
    When tracking, rejects detections below last position or too
    far horizontally from launch.
    """
    half = crop_size // 2
    if last_pos is not None:
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

    best = None
    best_score = 0

    for result in results:
        boxes = result.boxes
        for i in range(len(boxes)):
            c = float(boxes.conf[i])
            bx1, by1, bx2, by2 = map(int, boxes.xyxy[i])
            det_cx = (bx1 + bx2) // 2 + x1
            det_cy = (by1 + by2) // 2 + y1
            det_w = bx2 - bx1
            det_h = by2 - by1

            score = c

            if last_pos is not None:
                if det_cy > last_pos[1] + 30:
                    continue
                dist = np.sqrt((det_cx - cx) ** 2 + (det_cy - cy) ** 2)
                score += max(0, 1.0 - dist / half) * 0.5

            if launch_pos is not None:
                if abs(det_cx - launch_pos[0]) > full_w * 0.4:
                    continue

            if score > best_score:
                best_score = score
                best = (det_cx, det_cy, det_w, det_h, c)

    return best, (x1, y1, x2, y2)


def is_upward_trajectory(real_detections, launch_y, min_real=3, min_rise=80):
    """
    Only counts real YOLO detections (not Kalman coasted positions).
    Requires:
      - At least min_real actual detections
      - Ball moved upward by at least min_rise px
      - Detections are spread out (not all clustered on the tee)
    """
    if len(real_detections) < min_real or launch_y is None:
        return False

    ys = [p[1] for p in real_detections]
    min_y = min(ys)
    rise = launch_y - min_y

    if rise < min_rise:
        return False

    # Check that detections are spread vertically — not all in the
    # same spot (which would mean ball sitting on tee being re-detected)
    y_spread = max(ys) - min(ys)
    if y_spread < min_rise * 0.5:
        return False

    return True


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
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        print(f"Error: {video_path} not found")
        sys.exit(1)

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        print("Train first: python train.py")
        sys.exit(1)

    print(f"Loading model: {model_path}")
    model = YOLO(str(model_path))

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {w}x{h} @ {fps:.1f} fps, {total} frames")

    # Skip frames if input FPS exceeds target
    frame_skip = max(1, int(round(fps / args.target_fps))) if fps > args.target_fps else 1
    out_fps = fps / frame_skip
    if frame_skip > 1:
        print(f"Input {fps:.0f}fps > target {args.target_fps}fps — "
              f"processing every {frame_skip} frame(s), output at {out_fps:.1f}fps")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(args.output, fourcc, out_fps, (w, h))

    # ── Search zone: central 30% width, bottom 50% height ──
    search_x1 = int(w * 0.35)
    search_x2 = int(w * 0.65)
    search_y1 = int(h * 0.50)
    search_y2 = int(h * 0.95)
    print(f"Search zone: ({search_x1},{search_y1}) to ({search_x2},{search_y2})")

    INIT_CONF = max(args.conf, 0.5)

    kf = None
    all_trails = []          # list of (start_frame, end_frame, trail_points)
    current_trail = []
    real_detections = []     # only positions from actual YOLO hits
    launch_y = None
    moving_up = False        # set once ball is detected above launch point
    moving_up_frame = None   # frame when upward motion first detected

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        # Skip frames for high-FPS videos
        if frame_skip > 1 and (frame_idx % frame_skip) != 0:
            continue

        detection = None
        crop_rect = None

        if kf is None:
            # ── Search only in the restricted zone ──
            search_crop = frame[search_y1:search_y2, search_x1:search_x2]
            results = model(search_crop, conf=INIT_CONF, verbose=False)

            best_conf = 0
            for result in results:
                boxes = result.boxes
                for i in range(len(boxes)):
                    c = float(boxes.conf[i])
                    if c > best_conf:
                        bx1, by1, bx2, by2 = map(int, boxes.xyxy[i])
                        cx = (bx1 + bx2) // 2 + search_x1
                        cy = (by1 + by2) // 2 + search_y1
                        bw = bx2 - bx1
                        bh = by2 - by1
                        best_conf = c
                        detection = (cx, cy, bw, bh, c)

            if detection is not None:
                cx, cy = detection[0], detection[1]
                kf = BallKalmanFilter(cx, cy)
                current_trail = [(cx, cy)]
                launch_y = cy
                print(f"  Frame {frame_idx}: Ball found at ({cx},{cy}) "
                      f"conf={detection[4]:.2f}")

        else:
            # ── Track: predict, crop, detect, update ──
            predicted = kf.predict()
            pred_x, pred_y = int(predicted[0]), int(predicted[1])

            speed = kf.speed
            crop = args.crop_size
            if speed > 30:
                crop = int(crop * 1.5)
            if kf.missed > 3:
                crop = int(crop * (1 + kf.missed * 0.2))
            crop = min(crop, max(w, h))

            lp = current_trail[-1] if current_trail else None
            lnch = current_trail[0] if current_trail else None
            # Lower confidence when tracking
            track_conf = args.conf * 0.4  # e.g. 0.25 -> 0.10
            detection, crop_rect = detect_in_region(
                model, frame, pred_x, pred_y, crop, track_conf, w, h,
                last_pos=lp, launch_pos=lnch)

            if detection is not None:
                cx, cy = detection[0], detection[1]
                kf.update([cx, cy])
                current_trail.append((cx, cy))
                real_detections.append((cx, cy, frame_idx))
                if not moving_up and launch_y is not None and cy < launch_y - 20:
                    moving_up = True
                    moving_up_frame = frame_idx
            else:
                kf.mark_missed()
                px, py = kf.position
                if 0 <= px < w and 0 <= py < h:
                    current_trail.append((px, py))

            if kf.missed >= args.max_lost or (moving_up and kf.missed >= 10):
                if is_upward_trajectory(real_detections, launch_y):
                    # Trim spurious tail: remove trailing points that
                    # jump too far from the previous point
                    trimmed = list(real_detections)
                    while len(trimmed) > 2:
                        dx = trimmed[-1][0] - trimmed[-2][0]
                        dy = trimmed[-1][1] - trimmed[-2][1]
                        dist = np.sqrt(dx**2 + dy**2)
                        if dist > 50:
                            trimmed.pop()
                        else:
                            break
                    start_f = moving_up_frame if moving_up_frame else frame_idx
                    all_trails.append((start_f, frame_idx, trimmed))
                    print(f"  Frame {frame_idx}: Shot captured! "
                          f"{len(real_detections)} real detections")
                else:
                    print(f"  Frame {frame_idx}: Discarded trail "
                          f"({len(current_trail)} pts, "
                          f"{len(real_detections)} real) — not upward")
                kf = None
                current_trail = []
                real_detections = []
                launch_y = None
                moving_up = False
                moving_up_frame = None

        # ── Draw ──

        if args.debug:
            if kf is None:
                cv2.rectangle(frame, (search_x1, search_y1),
                              (search_x2, search_y2), (255, 0, 0), 2)
                cv2.putText(frame, "SEARCH ZONE", (search_x1, search_y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            if crop_rect is not None:
                rx1, ry1, rx2, ry2 = crop_rect
                cv2.rectangle(frame, (rx1, ry1), (rx2, ry2),
                              (255, 100, 0), 1)
            if kf is not None:
                px, py = kf.position
                cv2.drawMarker(frame, (px, py), (0, 100, 255),
                               cv2.MARKER_CROSS, 15, 1)

        if frame_idx % 50 == 0:
            st = "tracking" if kf else "searching"
            print(f"  Frame {frame_idx}/{total} [{st}] "
                  f"trail={len(current_trail)} shots={len(all_trails)}")

    # Check if there's a valid trail still being tracked at end of video
    if real_detections and is_upward_trajectory(real_detections, launch_y):
        start_f = moving_up_frame if moving_up_frame else frame_idx
        all_trails.append((start_f, frame_idx, list(real_detections)))

    cap.release()

    # Select trails to draw
    if all_trails:
        if args.all_trails:
            draw_trails = all_trails
            print(f"\nDrawing all {len(all_trails)} trail(s)")
        else:
            draw_trails = [max(all_trails, key=lambda t: len(t[2]))]
            print(f"\nBest shot: {len(draw_trails[0][2])} points "
                  f"(frames {draw_trails[0][0]}–{draw_trails[0][1]}, "
                  f"from {len(all_trails)} candidate(s))")
        for i, (s, e, t) in enumerate(all_trails):
            print(f"  Trail {i+1}: {len(t)} points (frames {s}–{e})")
    else:
        draw_trails = []
        print("\nNo valid upward trajectories found.")

    # ── Second pass: write output with trails at the right time ──
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
        for _, _, trail in draw_trails:
            pts = [(x, y) for x, y, f in trail if f <= frame_idx]
            if len(pts) > 1:
                for i in range(1, len(pts)):
                    cv2.line(overlay, pts[i-1], pts[i],
                             (0, 0, 255), 30, cv2.LINE_AA)
                drew = True
        if drew:
            cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

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
