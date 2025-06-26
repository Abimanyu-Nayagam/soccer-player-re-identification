from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import cv2
import torchreid
import sys

# Load a pre-trained YOLOv8 model
model = YOLO('./models/best.pt')
video_path = './Assignment Materials/Assignment Materials/15sec_input_720p.mp4'

feature_model = torchreid.models.build_model(
    name='osnet_x0_25',
    num_classes=1000,
    pretrained=True
)
model.eval().to('cuda')

tracker = DeepSort(max_age=5, n_init=10, max_cosine_distance=0.8, half=True, bgr=True, embedder="torchreid", embedder_model_name="osnet_ain_x1_0")

# Check if the model is loaded correctly
if model is None:
    print("Error: Model not loaded.")
    sys.exit(1)

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

frame_count = 0

while True:
    frame_count += 1
    ret, frame = cap.read()
    if not ret:
        print("End of video or error reading frame.")
        break
    
    if frame_count == 1 or frame_count % 3 != 0:
        results = model(frame)
        boxes = results[0].boxes if results else None

        detections = []

        print("Gathering detections...")
        if boxes is None or len(boxes) == 0:
            print("No detections found.")
            continue
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            cls = box.cls[0]
            if cls != 2:
                continue
            conf = box.conf[0]
            detections.append([(x1, y1, w, h), cls, conf])
        
        # Update Deep SORT tracker
    tracks = tracker.update_tracks(detections, frame=frame)

    # Draw tracks
    for i, track in enumerate(tracks):
        if not track.is_confirmed():
            continue
        ltrb = track.to_ltrb()
        print(f"ltrb: {ltrb}")
        track_id = track.track_id

        if len(previous_tracks) > 0:
            # Compare with all previous tracks to identify if the current track is a continuation of a previous one
            for j, prev_track in enumerate(previous_tracks):
                comp_x1, comp_y1, comp_x2, comp_y2 = prev_track.to_ltrb()
                # Calculate the center of the top edge of the previous track and the current track
                prev_center_x = (comp_x1 + comp_x2) / 2
                prev_center_y = comp_y1
                current_center_x = (ltrb[0] + ltrb[2]) / 2
                current_center_y = ltrb[1]
                # Assign previous track ID if the current track is close enough to a previous one
                if abs(prev_center_x - current_center_x) < 50 and abs(prev_center_y - current_center_y) < 50:
                    track_id =  prev_track.track_id
                    # Remove the previous track from the list to avoid reusing it
                    previous_tracks.pop(j)

        cv2.rectangle(frame, (int(ltrb[0]), int(ltrb[1])), (int(
            ltrb[2]), int(ltrb[3])), (0, 0, 255), 2)
        cv2.putText(frame, f"{str(track_id)}", (int(ltrb[0]), int(
            ltrb[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    previous_tracks = tracks.copy()

    cv2.imshow('YOLOv11 Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Exiting video stream.")
        break


cv2.destroyAllWindows()
cap.release()

print("Detections gathered, tracking...")