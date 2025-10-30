import cv2
from ultralytics import YOLO
import os
from vidgear.gears import CamGear

# ==== CONFIG ====
test_url = "https://www.youtube.com/watch?v=qhVWl2Xlq0I"
model_path = os.path.join(
    "D:/StudyRelated/Machine Learning Projects/NBA/dataset/yolo_training",
    "small_model", "weights", "best.pt"
)
output_path = "D:/StudyRelated/Machine Learning Projects/NBA/detections_output.mp4"

threshold = 0.45

# Class names and colours (BGR)
CLASS_NAMES = ["USA Player", "Opponent Player", "Basketball", "Referee"]
CLASS_COLORS = {
    0: (0, 0, 255),     # Red for USA Player
    1: (0, 255, 0),     # Green for Opponent
    2: (255, 165, 0),   # Orange for Basketball
    3: (255, 0, 255)    # Magenta for Referee
}

# ==== LOAD MODEL ====
model = YOLO(model_path)

# ==== OPEN VIDEO STREAM ====
stream = CamGear(source=test_url, stream_mode=True, logging=True).start()

# Grab first frame to set up writer
frame = stream.read()
if frame is None:
    raise RuntimeError("Could not read video stream")

height, width = frame.shape[:2]
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
fps = 30  # YouTube stream ~30fps
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

print(f"🔄 Processing video... saving to {output_path}")

while True:
    frame = stream.read()
    if frame is None:
        break

    results = model(frame)[0]

    # Loop through detections
    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result
        if score < threshold:
            continue

        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
        class_id = int(class_id)
        color = CLASS_COLORS.get(class_id, (200, 200, 0))
        label = CLASS_NAMES[class_id]

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{label} {score:.2f}",
                    (x1, max(15, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Write annotated frame to output video
    out.write(frame)

# ==== CLEANUP ====
stream.stop()
out.release()
print("✅ Done! Saved annotated video to:", output_path)
