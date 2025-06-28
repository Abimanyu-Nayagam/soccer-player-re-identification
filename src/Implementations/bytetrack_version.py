import sys
from pathlib import Path
from ultralytics import YOLO
import cv2
import supervision as sv
import numpy as np
# Resolve the root of the project relative to this file's location
ROOT_DIR = Path(__file__).resolve().parents[2]  # 2 levels up from implementations/

# Add the project root to sys.path
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Now you can import from configs/
from configs.config import MODEL_PATH, VIDEO_PATH

# Load a pre-trained YOLOv8 model
model = YOLO(MODEL_PATH)

# Initialize ByteTrack tracker and annotators
tracker = sv.ByteTrack()
box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()
trace_annotator = sv.TraceAnnotator()

def callback(frame: np.ndarray, _: int) -> np.ndarray:
    results = model(frame)[0]
    detections = sv.Detections.from_ultralytics(results)
    detections = tracker.update_with_detections(detections)

    # Create labels with class names and tracker IDs
    labels = [
        f"#{tracker_id} {results.names[class_id]}"
        for class_id, tracker_id
        in zip(detections.class_id, detections.tracker_id)
    ]

    annotated_frame = box_annotator.annotate(
        frame.copy(), detections=detections)
    res =  label_annotator.annotate(
        annotated_frame, detections=detections, labels=labels)
    return res

sv.process_video(
    source_path=VIDEO_PATH,
    target_path="./outputs/bytetrack_result.mp4",
    callback=callback
)
