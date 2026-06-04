import cv2
from ultralytics import YOLO
import os
from collections import defaultdict, deque
from vidgear.gears import CamGear

# CONFIG
test_url = "https://www.youtube.com/watch?v=qhVWl2Xlq0I"
model_path = os.path.join(
    "D:/StudyRelated/Machine Learning Projects/NBA/dataset/yolo_training",
    "small_model", "weights", "best.pt"
)
output_path = "D:/StudyRelated/Machine Learning Projects/NBA/detections_tracked.mp4"

threshold = 0.45
TRAIL_LEN = 30  # how many past centroids to draw per ID

CLASS_NAMES = ["USA Player", "Opponent Player", "Basketball", "Referee"]
CLASS_COLORS = {
    0: (0, 0, 255),
    1: (0, 255, 0),
    2: (255, 165, 0),
    3: (255, 0, 255),
}

model = YOLO(model_path)

stream = CamGear(source=test_url, stream_mode=True, logging=True).start()
frame = stream.read()
if frame is None:
    raise RuntimeError("Could not read video stream")

height, width = frame.shape[:2]
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
fps = 30
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# Per-ID centroid history for trail visualisation
trails = defaultdict(lambda: deque(maxlen=TRAIL_LEN))

print(f"Processing with ByteTrack tracking, saving to {output_path}")

while True:
    frame = stream.read()
    if frame is None:
        break

    # ByteTrack via Ultralytics built-in tracker.
    # persist=True keeps ID state across calls; tracker yaml selects ByteTrack.
    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        conf=threshold,
        verbose=False,
    )[0]

    boxes = results.boxes
    if boxes is None or boxes.id is None:
        out.write(frame)
        continue

    xyxy = boxes.xyxy.cpu().numpy()
    ids = boxes.id.int().cpu().numpy()
    cls = boxes.cls.int().cpu().numpy()
    conf = boxes.conf.cpu().numpy()

    for (x1, y1, x2, y2), tid, c, s in zip(xyxy, ids, cls, conf):
        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
        colour = CLASS_COLORS.get(int(c), (200, 200, 0))
        label = f"{CLASS_NAMES[int(c)]} #{int(tid)} {s:.2f}"

        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        trails[int(tid)].append((cx, cy))

        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
        cv2.putText(frame, label, (x1, max(15, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2)

        pts = list(trails[int(tid)])
        for i in range(1, len(pts)):
            cv2.line(frame, pts[i - 1], pts[i], colour, 2)

    out.write(frame)

stream.stop()
out.release()
print("Done. Saved tracked video to:", output_path)
