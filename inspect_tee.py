"""
Dump the tee zone at multiple frames so we can see what the ball
looks like and what brightness/colour values it has.
"""
import sys
import cv2
import numpy as np

video = sys.argv[1]
cap = cv2.VideoCapture(video)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Tee zone
tx1 = int(w * 0.20)
tx2 = int(w * 0.80)
ty1 = int(h * 0.65)
ty2 = int(h * 0.90)

# Sample frames: early, middle
sample_frames = [15, 30, 45, 60, 90]

for target in sample_frames:
    cap.set(cv2.CAP_PROP_POS_FRAMES, target)
    ret, frame = cap.read()
    if not ret:
        continue

    crop = frame[ty1:ty2, tx1:tx2]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    # Show what different thresholds pick up
    for thresh in [150, 170, 190, 210]:
        _, bright = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)
        cv2.imwrite(f"tee_f{target}_thresh{thresh}.png", bright)

    # Save the raw crop and grayscale
    cv2.imwrite(f"tee_f{target}_color.png", crop)
    cv2.imwrite(f"tee_f{target}_gray.png", gray)

    # Save HSV value channel (brightness)
    v_channel = hsv[:, :, 2]
    cv2.imwrite(f"tee_f{target}_value.png", v_channel)

    # Print stats about the brightest spots
    print(f"\nFrame {target}:")
    print(f"  Crop size: {crop.shape}")
    print(f"  Gray min={gray.min()} max={gray.max()} mean={gray.mean():.0f}")
    print(f"  V-chan min={v_channel.min()} max={v_channel.max()} mean={v_channel.mean():.0f}")

    # Find the brightest small region
    blurred = cv2.GaussianBlur(gray, (11, 11), 0)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(blurred)
    print(f"  Brightest spot: val={max_val:.0f} at {max_loc} (in crop coords)")
    print(f"  Full frame coords: ({max_loc[0]+tx1}, {max_loc[1]+ty1})")

cap.release()
print(f"\nSaved tee zone images. Look at tee_f*_color.png to find the ball.")
print("Check which threshold image (150/170/190/210) shows the ball as white.")
