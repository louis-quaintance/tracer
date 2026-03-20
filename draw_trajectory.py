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
    Fit x(t) and y(t) separately using frame number as the time parameter,
    then extrapolate forward until the ball returns to ground level.

    x(t): linear with perspective deceleration — ball moves into the
           distance so apparent speed decreases.
    y(t): quadratic — gravity pulls the ball back down.

    The ball's velocity at the last detection dictates how far and high
    the projected arc extends.

    Returns a list of (x, y) image-coordinate points for the full arc.
    """
    frames = np.array([p[0] for p in launch_points], dtype=np.float64)
    xs = np.array([p[1] for p in launch_points], dtype=np.float64)
    ys = np.array([p[2] for p in launch_points], dtype=np.float64)

    # Normalise time to start at 0
    t = frames - frames[0]

    # --- Fit x(t): linear velocity from detections ---
    x_coeffs = np.polyfit(t, xs, 1)  # x(t) = vx*t + x0
    vx = x_coeffs[0]  # pixels per frame in x
    x0 = x_coeffs[1]

    # --- Fit y(t): linear velocity from detections (rising phase only) ---
    y_lin = np.polyfit(t, ys, 1)
    vy = y_lin[0]   # negative in image coords = ball going up
    y0 = y_lin[1]

    # Ground level = y of first detection (where the ball was hit from)
    ground_y = ys[0]

    # Total rise observed so far (in pixels)
    rise_px = ground_y - ys[-1]  # positive = ball went up

    # --- Model a golf-shot-like arc ---
    # A real golf shot: the ball rises steeply at first, reaches apex
    # roughly 1/3 to 1/2 of the way through the flight, then descends
    # on a longer, shallower path back to ground.
    #
    # We know the initial vertical velocity (vy, negative = up in image).
    # Time to apex: t_apex where vy + g*t_apex = 0 => t_apex = -vy/g
    # We want the apex to be high and the flight to be LONG.
    #
    # Estimate: the ball has been rising for t[-1] frames and is still
    # going up strongly, so apex is well beyond what we've seen.
    # Use the velocity to estimate time-to-apex as a multiple of what
    # we've observed.

    t_observed = t[-1] if t[-1] > 0 else 1

    # --- Model a golf-shot arc with perspective ---
    # The ball rises steeply near the camera then travels far into the
    # distance. In image coordinates:
    #   - The apex is high up in the frame
    #   - The descent converges toward the horizon (landing y is much
    #     HIGHER in the image than launch y, because the landing spot
    #     is far away in perspective)
    #   - Apparent speed decreases as the ball gets further away

    # Apex at ~4x the observed rising time
    t_apex = t_observed * 4

    # Gravity to match: g = -vy / t_apex (vy is negative = upward)
    g = -vy / t_apex

    # Total flight: ball takes longer to descend than rise (longer arc)
    t_land = t_apex * 2.5

    # Ensure flight is substantially longer than what we observed
    t_land = max(t_land, t_observed * 10)

    print(f"  Velocity: vx={vx:.1f} vy={vy:.1f} px/frame")
    print(f"  Estimated apex at t={t_apex:.0f}, landing at t={t_land:.0f} "
          f"(observed {t_observed:.0f} frames)")

    # --- Landing height (perspective) ---
    # The ball lands far away, so in the image it lands much higher up
    # than where it was hit. The landing y should be somewhere between
    # the apex and the launch height — roughly 40-60% of the way up
    # from launch to apex.
    y_at_apex = y0 + vy * t_apex + 0.5 * g * t_apex**2
    apex_rise = ground_y - y_at_apex  # how far above ground the apex is
    # Landing point in image is well above ground (far away in perspective)
    landing_y = ground_y - apex_rise * 0.55

    # --- Generate the full arc ---
    n_pts = 800
    t_curve = np.linspace(0, t_land, n_pts)

    # Perspective scale: things further in time are further in distance,
    # so both x-movement and y-movement shrink. This factor goes from
    # 1.0 (at launch) toward ~0 (at landing, far in distance).
    # Use a smooth function that decelerates strongly toward the end.
    perspective = 1.0 / (1.0 + 0.02 * t_curve)

    # x(t): horizontal movement with strong perspective deceleration.
    # Ball keeps moving in x but progressively slower — converging
    # toward a distant point.
    curve_xs = x0 + vx * t_curve * perspective

    # y(t): parabolic arc in "world space", then compress the descent
    # side with perspective so the ball lands high up in the image.
    #
    # Raw parabola (no perspective):
    y_raw = y0 + vy * t_curve + 0.5 * g * t_curve**2
    #
    # But the descent side needs to land at landing_y, not ground_y.
    # Blend: before apex, use raw parabola; after apex, interpolate
    # between the raw parabola and the perspective-adjusted landing.
    curve_ys = np.copy(y_raw)

    t_apex_actual = -vy / g if g > 0 else t_apex
    y_apex = float(y_raw[np.argmin(y_raw)])
    for i, tt in enumerate(t_curve):
        if tt > t_apex_actual:
            # How far through the descent are we (0=apex, 1=landing)
            descent_progress = (tt - t_apex_actual) / max(t_land - t_apex_actual, 1)
            descent_progress = min(descent_progress, 1.0)

            # Very high exponent so ball hangs near apex for most of
            # the descent, only dropping in the final ~15%.
            ease = descent_progress ** 17
            curve_ys[i] = y_apex + (landing_y - y_apex) * ease

    # Build trajectory, stop at frame bounds
    trajectory = []
    for cx, cy in zip(curve_xs, curve_ys):
        ix, iy = int(round(cx)), int(round(cy))
        if 0 <= ix < frame_w and 0 <= iy < frame_h:
            trajectory.append((ix, iy))

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


def draw_line_on_frame(frame, points, thickness=5, color=(0, 0, 255)):
    """Draw a smooth line through a list of (x, y) points."""
    if len(points) < 2:
        return frame

    for i in range(len(points) - 1):
        # Slight gradient: brighter at the start
        t = i / max(len(points) - 1, 1)
        r = int(255 - 60 * t)
        col = (int(30 * t), int(20 * t), r)  # BGR
        cv2.line(frame, points[i], points[i + 1], col, thickness, cv2.LINE_AA)

    return frame


def draw_trajectory_on_frame(frame, trajectory_points, thickness=6):
    """Draw a smooth gradient trajectory arc on the frame."""
    if len(trajectory_points) < 2:
        return frame

    n = len(trajectory_points)
    for i in range(n - 1):
        t = i / max(n - 1, 1)
        r = int(255 - 80 * t)
        g = int(30 * t)
        b_val = int(30 * t)
        color = (b_val, g, r)  # BGR

        pt1 = trajectory_points[i]
        pt2 = trajectory_points[i + 1]
        cv2.line(frame, pt1, pt2, color, thickness, cv2.LINE_AA)

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
    parser.add_argument("--thickness", type=int, default=6,
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
