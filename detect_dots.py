"""
Draw red dots wherever the model detects a golf ball in every frame.

Usage:
    python detect_dots.py <video_path> --model best_v2.pt [--conf 0.25] [--show]
"""

import argparse
import sys
from pathlib import Path

import cv2
from ultralytics import YOLO


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

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        results = model(frame, conf=args.conf, verbose=False)

        for result in results:
            boxes = result.boxes
            for i in range(len(boxes)):
                bx1, by1, bx2, by2 = map(int, boxes.xyxy[i])
                cx = (bx1 + bx2) // 2
                cy = (by1 + by2) // 2
                conf = float(boxes.conf[i])
                cv2.circle(frame, (cx, cy), 12, (0, 0, 255), -1)
                cv2.circle(frame, (cx, cy), 12, (0, 0, 180), 2)
                detections_total += 1

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
