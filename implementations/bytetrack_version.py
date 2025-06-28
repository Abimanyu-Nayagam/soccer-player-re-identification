from ultralytics import YOLO
import cv2
import supervision as sv
import numpy as np

# Load a pre-trained YOLOv8 model
model = YOLO('./models/best.pt')
video_path = './Assignment Materials/Assignment Materials/15sec_input_720p.mp4'

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
    cv2.imshow("Frame", res)
    key = cv2.waitKey(0)  # Wait for key press before continuing
    if key == ord('q'):
        # Optional: quit entire process (not just frame skipping)
        print("Early exit triggered.")
        exit()
    return res

sv.process_video(
    source_path=video_path,
    target_path="./outputs/bytetrack_result.mp4",
    callback=callback
)
