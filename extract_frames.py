"""
Extract frames from videos for annotation.

Usage:
    python extract_frames.py videos/IMG_0572.MOV [videos/IMG_0573.MOV ...]
    python extract_frames.py videos/*.MOV --every 10 --output frames
"""

import argparse
import sys
from pathlib import Path

import cv2


def main():
    parser = argparse.ArgumentParser(description="Extract frames from videos for annotation")
    parser.add_argument("videos", nargs="+", help="Video files to extract from")
    parser.add_argument("--output", "-o", default="frames", help="Output directory")
    parser.add_argument("--every", type=int, default=5,
                        help="Extract every Nth frame (default: 5)")
    parser.add_argument("--max-per-video", type=int, default=200,
                        help="Max frames to extract per video (default: 200)")
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    total_saved = 0

    for video_path in args.videos:
        video_path = Path(video_path)
        if not video_path.exists():
            print(f"Skipping {video_path} — not found")
            continue

        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        stem = video_path.stem

        print(f"\n{video_path.name}: {w}x{h} @ {fps:.0f}fps, {frame_count} frames")

        frame_idx = 0
        saved = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1

            if frame_idx % args.every != 0:
                continue

            if saved >= args.max_per_video:
                break

            filename = f"{stem}_{frame_idx:05d}.jpg"
            cv2.imwrite(str(out_dir / filename), frame)
            saved += 1

        cap.release()
        total_saved += saved
        print(f"  Saved {saved} frames")

    print(f"\nDone! {total_saved} frames saved to {out_dir}/")
    print(f"\nNext steps:")
    print(f"  1. pip install labelImg")
    print(f"  2. labelImg {out_dir}/")
    print(f"  3. Draw boxes around golf balls, save as PascalVOC")
    print(f"     - Set class name to 'golfball'")
    print(f"     - Annotations will be saved as XML next to the images")


if __name__ == "__main__":
    main()
