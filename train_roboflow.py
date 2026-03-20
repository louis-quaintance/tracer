"""
Train YOLOv8 on the Roboflow golf-ball dataset.

Converts Pascal VOC annotations from roboflow_data/{train,valid,test}
to YOLO format, then trains.

Usage:
    python train_roboflow.py [--epochs 100] [--model yolov8m.pt] [--imgsz 1280]
"""

import argparse
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

from ultralytics import YOLO

PROJECT_DIR = Path(__file__).parent
ROBOFLOW_DIR = PROJECT_DIR / "roboflow_data"
YOLO_DATASET = PROJECT_DIR / "dataset_yolo_roboflow"

TARGET_CLASSES = {"golf-ball", "golfball"}
MIN_BOX_PX = 3  # ignore boxes smaller than 3px — likely annotation noise


def convert_voc_image(xml_path, img_path, img_out, lbl_out):
    """Convert a single VOC annotation + image to YOLO format.

    Returns a tuple (converted: bool, skipped_annotations: int).
    """
    if not xml_path.exists() or not img_path.exists():
        return False, 0

    try:
        tree = ET.parse(xml_path)
    except ET.ParseError:
        return False, 0

    root = tree.getroot()

    size = root.find("size")
    if size is None:
        return False, 0

    w_node = size.find("width")
    h_node = size.find("height")
    if w_node is None or h_node is None:
        return False, 0

    img_w = int(w_node.text)
    img_h = int(h_node.text)
    if img_w == 0 or img_h == 0:
        return False, 0

    labels = []
    skipped = 0
    has_golf_objects = False
    for obj in root.findall("object"):
        name_node = obj.find("name")
        if name_node is None or name_node.text is None:
            skipped += 1
            continue

        if name_node.text.strip().lower() not in TARGET_CLASSES:
            continue

        has_golf_objects = True

        bbox = obj.find("bndbox")
        if bbox is None:
            skipped += 1
            continue

        try:
            xmin = float(bbox.find("xmin").text)
            ymin = float(bbox.find("ymin").text)
            xmax = float(bbox.find("xmax").text)
            ymax = float(bbox.find("ymax").text)
        except (AttributeError, TypeError, ValueError):
            skipped += 1
            continue

        # Clip raw coordinates to image bounds first
        xmin = max(0.0, min(xmin, img_w))
        xmax = max(0.0, min(xmax, img_w))
        ymin = max(0.0, min(ymin, img_h))
        ymax = max(0.0, min(ymax, img_h))

        if xmax <= xmin or ymax <= ymin:
            skipped += 1
            continue

        # Skip degenerate tiny boxes (likely annotation noise)
        if (xmax - xmin) < MIN_BOX_PX or (ymax - ymin) < MIN_BOX_PX:
            skipped += 1
            continue

        cx = (xmin + xmax) / 2.0 / img_w
        cy = (ymin + ymax) / 2.0 / img_h
        bw = (xmax - xmin) / img_w
        bh = (ymax - ymin) / img_h

        labels.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

    if not labels and not has_golf_objects:
        return False, skipped

    out_name = img_path.stem
    # Preserve original extension instead of blindly renaming to .jpg
    out_img = img_out / f"{out_name}{img_path.suffix.lower()}"
    shutil.copy2(img_path, out_img)
    # Write label file (empty for negative images — teaches model to avoid FPs)
    (lbl_out / f"{out_name}.txt").write_text("\n".join(labels) + "\n" if labels else "")
    return True, skipped


def convert_split(split_src, split_name, img_out, lbl_out):
    """Convert all VOC images in a roboflow split directory."""
    if not split_src.exists():
        print(f"  WARNING: {split_src} does not exist, skipping")
        return

    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    converted = 0
    negatives = 0
    skipped_images = 0
    skipped_annotations = 0
    xmls_without_labels = 0

    for xml_path in sorted(split_src.glob("*.xml")):
        stem = xml_path.stem
        # Find the matching image — try common extensions
        img_path = None
        for ext in (".jpg", ".jpeg", ".png", ".bmp"):
            candidate = split_src / f"{stem}{ext}"
            if candidate.exists():
                img_path = candidate
                break

        if img_path is None:
            skipped_images += 1
            continue

        ok, ann_skipped = convert_voc_image(xml_path, img_path, img_out, lbl_out)
        skipped_annotations += ann_skipped
        if ok:
            # Check if it was a negative (empty label file)
            lbl_file = lbl_out / f"{img_path.stem}.txt"
            if lbl_file.exists() and lbl_file.stat().st_size == 0:
                negatives += 1
            else:
                converted += 1
        else:
            if ann_skipped == 0:
                xmls_without_labels += 1

    print(f"  {split_name}: {converted} images with labels, {negatives} negative images")
    if skipped_images:
        print(f"    {skipped_images} XMLs with no matching image (skipped)")
    if xmls_without_labels:
        print(f"    {xmls_without_labels} XMLs with no valid golf-ball labels")
    if skipped_annotations:
        print(f"    {skipped_annotations} individual annotations skipped (degenerate/tiny boxes)")


def convert_all():
    """Convert the Roboflow dataset from Pascal VOC to YOLO format."""
    print("Converting Roboflow VOC annotations to YOLO format...")

    # Clean previous conversion
    if YOLO_DATASET.exists():
        shutil.rmtree(YOLO_DATASET)

    split_map = {
        "train": ROBOFLOW_DIR / "train",
        "val": ROBOFLOW_DIR / "valid",
        "test": ROBOFLOW_DIR / "test",
    }

    for split_name, split_src in split_map.items():
        img_out = YOLO_DATASET / "images" / split_name
        lbl_out = YOLO_DATASET / "labels" / split_name
        convert_split(split_src, split_name, img_out, lbl_out)

    yaml_path = YOLO_DATASET / "golfball.yaml"
    yaml_path.write_text(
        f"path: {YOLO_DATASET.resolve()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"test: images/test\n"
        f"\n"
        f"names:\n"
        f"  0: golfball\n"
    )
    print(f"\nDataset YAML: {yaml_path}")
    return yaml_path


def main():
    parser = argparse.ArgumentParser(description="Train golf ball detector on Roboflow data")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--model", default="yolov8m.pt",
                        help="Base model (default: yolov8m.pt — good balance for small objects)")
    parser.add_argument("--imgsz", type=int, default=1280,
                        help="Image size (default: 1280 — critical for tiny ~7px bboxes)")
    parser.add_argument("--batch", type=int, default=4,
                        help="Batch size (default: 4 — conservative for 1280 imgsz)")
    parser.add_argument("--freeze", type=int, default=0,
                        help="Freeze first N backbone layers (e.g. 10 for transfer learning)")
    parser.add_argument("--skip-convert", action="store_true",
                        help="Skip VOC-to-YOLO conversion if already done")
    args = parser.parse_args()

    if args.skip_convert and (YOLO_DATASET / "golfball.yaml").exists():
        yaml_path = YOLO_DATASET / "golfball.yaml"
        print(f"Skipping conversion, using {yaml_path}")
    else:
        yaml_path = convert_all()

    print(f"\n── Training YOLOv8 ──")
    print(f"  Base model: {args.model}")
    print(f"  Epochs:     {args.epochs}")
    print(f"  Image size: {args.imgsz}")
    print(f"  Batch size: {args.batch}")

    model = YOLO(args.model)

    train_kwargs = dict(
        data=str(yaml_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        name="golfball_roboflow",
        patience=25,
        save=True,
        plots=True,
        # ── Augmentation tuned for tiny objects (avg 7x6px on 640) ──
        mosaic=0.0,        # OFF — mosaic quarters the image, making 7px boxes ~2px
        scale=0.0,         # OFF — any downscale loses the ball entirely
        copy_paste=0.0,    # OFF — not useful for single-class tiny objects
        mixup=0.0,         # OFF — blending washes out tiny features
        flipud=0.0,        # no vertical flip (golf scenes have consistent orientation)
        fliplr=0.5,        # horizontal flip is fine
        degrees=3.0,       # very slight rotation (ball is round, scene matters)
        translate=0.1,     # slight shift
        hsv_h=0.015,       # hue shift
        hsv_s=0.4,         # saturation shift
        hsv_v=0.4,         # brightness shift (variable lighting)
        # ── Small-object specific ──
        close_mosaic=0,    # don't re-enable mosaic near end of training
    )

    if args.freeze > 0:
        train_kwargs["freeze"] = args.freeze
        print(f"  Freezing first {args.freeze} backbone layers")

    model.train(**train_kwargs)

    print("\n── Training complete! ──")
    print("Best model: runs/detect/golfball_roboflow/weights/best.pt")
    print("\nTo use it:")
    print("  python golf_tracer.py your_video.mp4 --model runs/detect/golfball_roboflow/weights/best.pt --show")


if __name__ == "__main__":
    main()
