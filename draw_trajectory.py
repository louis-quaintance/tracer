"""
Detect golf ball positions and draw a trajectory.

Two modes:
  1. Ball launched upward then LOST — project a full parabolic arc
     using the ball's velocity to determine how far into the distance
     it travels before landing.
  2. Ball tracked throughout (putting, chipping) — just trace a smooth
     line through all detected positions, no projection.

Usage:
    python draw_trajectory.py <video_path> --model best_new_v1.pt [--conf 0.25] [--show]
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


def collect_detections(model, cap, conf_threshold):
    """Run detection on every frame and return list of (frame_idx, cx, cy)."""
    detections = []
    frame_idx = 0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        results = model(frame, conf=conf_threshold, verbose=False)
        for result in results:
            boxes = result.boxes
            for i in range(len(boxes)):
                bx1, by1, bx2, by2 = map(int, boxes.xyxy[i])
                cx = (bx1 + bx2) // 2
                cy = (by1 + by2) // 2
                detections.append((frame_idx, cx, cy))

        if frame_idx % 50 == 0:
            print(f"  Scanning frame {frame_idx}/{total}")

    print(f"  Found {len(detections)} detections across {frame_idx} frames")
    return detections, frame_idx


def find_launch_sequence(detections, total_frames, min_points=4,
                         max_gap_frames=8, lost_threshold_frames=30):
    """
    Find an upward launch sequence where the ball is subsequently LOST.

    Returns the launch sequence if the ball is lost after it, else [].
    """
    if len(detections) < min_points:
        return []

    dets = sorted(detections, key=lambda d: d[0])

    # Build all candidate upward sequences
    all_sequences = []
    current_seq = [dets[0]]

    for i in range(1, len(dets)):
        prev_frame, prev_x, prev_y = current_seq[-1]
        cur_frame, cur_x, cur_y = dets[i]
        frame_gap = cur_frame - prev_frame

        if frame_gap <= max_gap_frames and cur_y < prev_y:
            current_seq.append(dets[i])
        else:
            if len(current_seq) >= min_points:
                all_sequences.append(current_seq[:])
            current_seq = [dets[i]]

    if len(current_seq) >= min_points:
        all_sequences.append(current_seq)

    if not all_sequences:
        return []

    for seq in sorted(all_sequences, key=len, reverse=True):
        seq_end_frame = seq[-1][0]

        window_after = lost_threshold_frames
        dets_after = [d for d in dets
                      if seq_end_frame < d[0] <= seq_end_frame + window_after]

        detection_density_after = len(dets_after) / max(window_after, 1)

        total_y_rise = seq[0][2] - seq[-1][2]
        frame_height_fraction = total_y_rise / max(1, seq[0][2])

        is_lost = detection_density_after < 0.15
        is_significant_rise = frame_height_fraction > 0.05

        print(f"  Candidate: {len(seq)} pts, frames {seq[0][0]}-{seq_end_frame}, "
              f"rise={total_y_rise}px ({frame_height_fraction:.0%}), "
              f"dets after={len(dets_after)}/{window_after} "
              f"(density={detection_density_after:.2f})")

        if is_lost and is_significant_rise:
            print(f"  -> Ball launched and LOST — will project trajectory")
            return seq
        else:
            reasons = []
            if not is_lost:
                reasons.append(f"ball still tracked ({len(dets_after)} dets "
                               f"in next {window_after} frames)")
            if not is_significant_rise:
                reasons.append(f"rise too small ({total_y_rise}px)")
            print(f"  -> Skipped: {'; '.join(reasons)}")

    return []


def fit_trajectory_parametric(launch_points, fps, frame_w, frame_h):
    """
    Build a trajectory from first detection to last, then project a
    parabolic arc from the last detected point into the distance.

    The projection barely moves in x (golf shot goes away from camera,
    not sideways). It arcs up, peaks, then comes back down to a landing
    point that's higher in the image (further away in perspective).
    """
    pts = sorted(launch_points, key=lambda p: p[0])
    frames = np.array([p[0] for p in pts], dtype=np.float64)
    xs = np.array([p[1] for p in pts], dtype=np.float64)
    ys = np.array([p[2] for p in pts], dtype=np.float64)

    # --- Detected portion: line through all captured points ---
    detected_line = [(int(x), int(y)) for x, y in zip(xs, ys)]

    # --- Estimate velocity from last few detections ---
    t = frames - frames[0]
    tail_n = min(4, len(t))
    vy = np.polyfit(t[-tail_n:], ys[-tail_n:], 1)[0]  # neg = going up

    last_x = float(xs[-1])
    last_y = float(ys[-1])
    ground_y = float(ys[0])
    rise_so_far = ground_y - last_y  # how far the ball has risen (px)

    print(f"  First detection: ({xs[0]:.0f}, {ys[0]:.0f})")
    print(f"  Last detection:  ({last_x:.0f}, {last_y:.0f})")
    print(f"  Rise so far: {rise_so_far:.0f}px, vy={vy:.1f} px/frame")

    # --- Projection: ballistic continuation from last detection ---
    # Uses the SAME velocity (vy) at the join point so the line continues
    # smoothly in the same direction, then gravity bends it into an arc.

    # Also estimate vx from the last few detections
    vx = np.polyfit(t[-tail_n:], xs[-tail_n:], 1)[0]

    # Time to apex: vy + g*t = 0. vy is negative (upward), g is positive.
    t_to_apex = max(8, abs(vy) * 3)
    g = abs(vy) / t_to_apex

    # Total flight time after last detection (descent a bit longer than rise)
    t_total = t_to_apex * 2.5

    # Landing y: ball lands far away, so higher in image (perspective)
    apex_y = last_y + vy * t_to_apex + 0.5 * g * t_to_apex**2
    total_peak_height = ground_y - apex_y
    landing_y = ground_y - total_peak_height * 0.4

    # --- Generate projection as a ballistic curve ---
    # y(dt) = last_y + vy*dt + 0.5*g*dt^2
    # This starts with exactly the same slope (vy) as the detected phase
    n_pts = 400
    dt = np.linspace(0, t_total, n_pts)

    # Raw ballistic y
    proj_ys = last_y + vy * dt + 0.5 * g * dt**2

    # After apex, blend descent toward landing_y (perspective compression)
    apex_idx = int(np.argmin(proj_ys))
    apex_y_actual = float(proj_ys[apex_idx])
    for i in range(apex_idx, n_pts):
        descent_t = (i - apex_idx) / max(n_pts - apex_idx, 1)
        # Raw ballistic wants to go back to ground, but perspective
        # means landing is higher up. Blend toward landing_y.
        raw = proj_ys[i]
        target = apex_y_actual + (landing_y - apex_y_actual) * (descent_t ** 2)
        proj_ys[i] = target

    # X: continue with the same vx, decelerating with perspective
    perspective = 1.0 / (1.0 + 0.1 * dt)
    proj_xs = last_x + vx * dt * perspective

    # Build projected points
    projected = []
    for cx, cy in zip(proj_xs[1:], proj_ys[1:]):
        ix, iy = int(round(cx)), int(round(cy))
        if 0 <= ix < frame_w and 0 <= iy < frame_h:
            projected.append((ix, iy))

    trajectory = detected_line + projected
    print(f"  Apex at y={apex_y:.0f}, landing at y={landing_y:.0f}")
    print(f"  Trajectory: {len(detected_line)} detected + "
          f"{len(projected)} projected points")
    return trajectory


def smooth_trace_line(detections):
    """
    Given all detections (possibly noisy), return a smoothed list of
    (x, y) points connecting them in frame order — for tracing the ball
    path without any projection.
    """
    dets = sorted(detections, key=lambda d: d[0])
    points = [(d[1], d[2]) for d in dets]

    if len(points) < 2:
        return points

    # Remove obvious outliers: points that jump too far from neighbours
    cleaned = [points[0]]
    for i in range(1, len(points)):
        dx = abs(points[i][0] - cleaned[-1][0])
        dy = abs(points[i][1] - cleaned[-1][1])
        dist = np.sqrt(dx**2 + dy**2)
        # Allow reasonable movement (up to 200px between detections)
        if dist < 200:
            cleaned.append(points[i])

    return cleaned


def draw_line_on_frame(frame, points, thickness=20, alpha=0.4):
    """Draw a thick semi-transparent red line through points."""
    if len(points) < 2:
        return frame

    overlay = frame.copy()
    for i in range(len(points) - 1):
        cv2.line(overlay, points[i], points[i + 1],
                 (0, 0, 255), thickness, cv2.LINE_AA)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    return frame


def draw_trajectory_on_frame(frame, trajectory_points, thickness=12, alpha=0.5):
    """Draw a thick semi-transparent red trajectory arc."""
    if len(trajectory_points) < 2:
        return frame

    overlay = frame.copy()
    for i in range(len(trajectory_points) - 1):
        cv2.line(overlay, trajectory_points[i], trajectory_points[i + 1],
                 (0, 0, 255), thickness, cv2.LINE_AA)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    return frame


def main():
    parser = argparse.ArgumentParser(
        description="Detect golf ball and draw trajectory / projected arc")
    parser.add_argument("video", help="Path to input video file")
    parser.add_argument("--model", default="best_new_v1.pt")
    parser.add_argument("--output", "-o", default="trajectory_output.mp4")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--min-points", type=int, default=4,
                        help="Minimum detections needed for a launch sequence")
    parser.add_argument("--thickness", type=int, default=17,
                        help="Line thickness for the trajectory")
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

    # --- Pass 1: collect all detections ---
    print("Pass 1: Detecting golf balls...")
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {w}x{h} @ {fps:.1f} fps, {total_frames} frames")

    detections, _ = collect_detections(model, cap, args.conf)
    cap.release()

    if not detections:
        print("No detections found — nothing to draw.")
        sys.exit(0)

    # --- Decide mode: projected arc vs simple trace ---
    print("Analysing ball movement...")
    launch = find_launch_sequence(detections, total_frames,
                                  min_points=args.min_points)

    trajectory = []       # projected arc points (mode 1)
    trace_points = []     # simple trace points (mode 2)
    launch_start_frame = 0
    arc_anim_frames = 1

    if launch:
        # MODE 1: Ball launched and lost — project full trajectory
        print(f"  Launch sequence: {len(launch)} points, "
              f"frames {launch[0][0]}-{launch[-1][0]}")
        for f, x, y in launch:
            print(f"    frame {f}: ({x}, {y})")

        print("Fitting projected trajectory...")
        trajectory = fit_trajectory_parametric(launch, fps, w, h)

        if len(trajectory) < 10:
            print("Could not fit a realistic trajectory — falling back to trace.")
            trajectory = []
            trace_points = smooth_trace_line(detections)
        else:
            print(f"  Projected arc: {len(trajectory)} points")
            launch_start_frame = launch[0][0]
            launch_end_frame = launch[-1][0]
            arc_anim_frames = max(30,
                                  (launch_end_frame - launch_start_frame) * 3)
    else:
        # MODE 2: Ball tracked throughout — just trace its path
        print("Ball tracked throughout — drawing trace line (no projection).")
        trace_points = smooth_trace_line(detections)
        print(f"  Trace: {len(trace_points)} points")

    # --- Pass 2: render output video ---
    print("Pass 2: Rendering output video...")
    cap = cv2.VideoCapture(str(video_path))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(args.output, fourcc, fps, (w, h))

    # For trace mode: build up the line progressively as detections appear
    dets_sorted = sorted(detections, key=lambda d: d[0])
    trace_idx = 0  # how many trace points to show so far

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        if trajectory:
            # MODE 1: Draw projected trajectory arc progressively
            # Small dots on detected positions
            for det_frame, cx, cy in detections:
                if det_frame == frame_idx:
                    cv2.circle(frame, (cx, cy), 6, (0, 0, 255), -1)

            if frame_idx >= launch_start_frame:
                progress = ((frame_idx - launch_start_frame) /
                            max(arc_anim_frames, 1))
                progress = min(progress, 1.0)
                n_pts = max(2, int(len(trajectory) * progress))
                draw_trajectory_on_frame(frame, trajectory[:n_pts],
                                         args.thickness)

        elif trace_points:
            # MODE 2: Draw trace line up to current frame
            # Advance trace index to include all detections up to this frame
            while (trace_idx < len(dets_sorted) and
                   dets_sorted[trace_idx][0] <= frame_idx):
                trace_idx += 1

            # Build the visible portion of the trace
            visible_trace = trace_points[:trace_idx]
            if len(visible_trace) >= 2:
                draw_line_on_frame(frame, visible_trace, args.thickness)

            # Dot on current detection
            for det_frame, cx, cy in detections:
                if det_frame == frame_idx:
                    cv2.circle(frame, (cx, cy), 8, (0, 0, 255), -1)
                    cv2.circle(frame, (cx, cy), 8, (0, 0, 180), 2)

        out.write(frame)

        if args.show:
            cv2.imshow("Trajectory", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        if frame_idx % 50 == 0:
            print(f"  Frame {frame_idx}/{total_frames}")

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"\nDone! Output: {args.output}")


if __name__ == "__main__":
    main()
