"""
Train YOLOv8 on golf ball detection + tracking datasets combined.

Converts Pascal VOC annotations from both data/Detection and data/Tracking
to YOLO format, then trains.

Usage:
    python train.py [--epochs 100] [--model yolov8s.pt] [--imgsz 640]
"""

import argparse
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

from ultralytics import YOLO


PROJECT_DIR = Path(__file__).parent
YOLO_DATASET = PROJECT_DIR / "dataset_yolo"


def convert_voc_image(xml_path, img_path, img_out, lbl_out, prefix=""):
    """Convert a single VOC annotation + image to YOLO format."""
    if not xml_path.exists() or not img_path.exists():
        return False

    tree = ET.parse(xml_path)
    root = tree.getroot()

    size = root.find("size")
    img_w = int(size.find("width").text)
    img_h = int(size.find("height").text)

    if img_w == 0 or img_h == 0:
        return False

    labels = []
    for obj in root.findall("object"):
        name = obj.find("name").text
        if name != "golfball":
            continue

        bbox = obj.find("bndbox")
        xmin = float(bbox.find("xmin").text)
        ymin = float(bbox.find("ymin").text)
        xmax = float(bbox.find("xmax").text)
        ymax = float(bbox.find("ymax").text)

        cx = max(0, min(1, (xmin + xmax) / 2.0 / img_w))
        cy = max(0, min(1, (ymin + ymax) / 2.0 / img_h))
        bw = max(0, min(1, (xmax - xmin) / img_w))
        bh = max(0, min(1, (ymax - ymin) / img_h))

        labels.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

    if not labels:
        return False

    out_name = f"{prefix}{img_path.stem}"
    shutil.copy2(img_path, img_out / f"{out_name}.jpg")
    (lbl_out / f"{out_name}.txt").write_text("\n".join(labels) + "\n")
    return True


def convert_detection_data(split_name, ids, img_out, lbl_out):
    """Convert Detection dataset (flat structure)."""
    det_root = PROJECT_DIR / "data" / "Detection"
    ann_dir = det_root / "Annotations"
    img_dir = det_root / "JPEGImages"

    converted = 0
    for img_id in ids:
        img_id = img_id.strip()
        if not img_id:
            continue
        xml_path = ann_dir / f"{img_id}.xml"
        img_path = img_dir / f"{img_id}.jpg"
        if convert_voc_image(xml_path, img_path, img_out, lbl_out, prefix="det_"):
            converted += 1

    return converted


def convert_tracking_data(split_name, img_out, lbl_out):
    """
    Convert Tracking dataset (sequence folders for images,
    flat annotations).
    """
    track_root = PROJECT_DIR / "data" / "Tracking"
    ann_dir = track_root / "Annotations"
    img_base = track_root / "JPEGImages"

    converted = 0

    # Build a map from annotation filename stem -> image path
    # Annotations are like 01_001.xml, images are in Golf_1/01_001.jpg etc.
    img_map = {}
    for seq_dir in img_base.iterdir():
        if not seq_dir.is_dir():
            continue
        for img_file in seq_dir.glob("*.jpg"):
            img_map[img_file.stem] = img_file

    for xml_file in sorted(ann_dir.glob("*.xml")):
        stem = xml_file.stem
        img_path = img_map.get(stem)
        if img_path is None:
            continue
        if convert_voc_image(xml_file, img_path, img_out, lbl_out, prefix="trk_"):
            converted += 1

    return converted


def convert_all():
    """Convert both Detection and Tracking datasets."""
    print("Converting VOC annotations to YOLO format...")

    det_root = PROJECT_DIR / "data" / "Detection"
    sets_dir = det_root / "ImageSets" / "Main"

    train_ids = (sets_dir / "train.txt").read_text().strip().split("\n")
    test_ids = (sets_dir / "test.txt").read_text().strip().split("\n")

    for split_name, ids in [("train", train_ids), ("val", test_ids)]:
        img_out = YOLO_DATASET / "images" / split_name
        lbl_out = YOLO_DATASET / "labels" / split_name
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        det_count = convert_detection_data(split_name, ids, img_out, lbl_out)
        print(f"  {split_name} Detection: {det_count} images")

        # Add tracking data to train set only (it's all one set)
        if split_name == "train":
            trk_count = convert_tracking_data(split_name, img_out, lbl_out)
            print(f"  {split_name} Tracking: {trk_count} images")

    yaml_path = YOLO_DATASET / "golfball.yaml"
    yaml_path.write_text(
        f"path: {YOLO_DATASET.resolve()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"\n"
        f"names:\n"
        f"  0: golfball\n"
    )
    print(f"\nDataset YAML: {yaml_path}")
    return yaml_path


def main():
    parser = argparse.ArgumentParser(description="Train golf ball detector")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--model", default="yolov8s.pt",
                        help="Base model (default: yolov8s.pt — better for small objects)")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Image size (default: 640 — higher res for small ball)")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--skip-convert", action="store_true")
    args = parser.parse_args()

    if args.skip_convert and (YOLO_DATASET / "golfball.yaml").exists():
        yaml_path = YOLO_DATASET / "golfball.yaml"
        print(f"Skipping conversion, using {yaml_path}")
    else:
        yaml_path = convert_all()

    print(f"\n── Training YOLOv8 ──")
    print(f"  Base model: {args.model}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Image size: {args.imgsz}")
    print(f"  Batch size: {args.batch}")

    model = YOLO(args.model)
    model.train(
        data=str(yaml_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        name="golfball",
        patience=20,
        save=True,
        plots=True,
        # Augmentation tuned for small object detection
        mosaic=1.0,
        scale=0.5,
        flipud=0.0,
        fliplr=0.5,
    )

    print("\n── Training complete! ──")
    print("Best model: runs/detect/golfball/weights/best.pt")
    print("\nTo use it:")
    print("  python golf_tracer.py your_video.mp4 --model runs/detect/golfball/weights/best.pt --show")


if __name__ == "__main__":
    main()
