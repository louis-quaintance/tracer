"""Save a frame from the video with grid overlay to help set zones."""
import sys
import cv2

video = sys.argv[1]
cap = cv2.VideoCapture(video)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Grab a frame ~1 second in
cap.set(cv2.CAP_PROP_POS_FRAMES, 30)
ret, frame = cap.read()
cap.release()

if not ret:
    print("Failed to read frame")
    sys.exit(1)

# Draw grid lines every 10% with labels
for pct in range(10, 100, 10):
    x = int(w * pct / 100)
    y = int(h * pct / 100)
    cv2.line(frame, (x, 0), (x, h), (100, 100, 100), 1)
    cv2.line(frame, (0, y), (w, y), (100, 100, 100), 1)
    cv2.putText(frame, f"{pct}%", (x + 2, 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
    cv2.putText(frame, f"{pct}%", (2, y - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)

# Pixel coords at edges
cv2.putText(frame, f"{w}x{h}", (w - 80, h - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

cv2.imwrite("snapshot.png", frame)
print(f"Saved snapshot.png ({w}x{h})")
print("Open it and tell me roughly where the ball sits (e.g. '50% across, 85% down')")
