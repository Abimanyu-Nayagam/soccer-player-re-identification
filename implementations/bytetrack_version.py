from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import cv2
import supervision as sv
import numpy as np
import torchreid

# Load a pre-trained YOLOv8 model
model = YOLO('../models/best.pt')
video_path = '../Assignment Materials/Assignment Materials/15sec_input_720p.mp4'


feature_model = torchreid.models.build_model(
    name='osnet_x0_25',
    num_classes=1000,
    pretrained=True
)
model.eval().to('cuda')

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
    annotated_frame = label_annotator.annotate(
        annotated_frame, detections=detections, labels=labels)
    return trace_annotator.annotate(
        annotated_frame, detections=detections)

sv.process_video(
    source_path=video_path,
    target_path="bytetrack_result.avi",
    callback=callback
)
