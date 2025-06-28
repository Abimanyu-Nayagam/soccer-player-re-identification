from pathlib import Path
import sys
import torch
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import cv2
from ultralytics import YOLO
from easyocr import Reader

# Resolve the root of the project relative to this file's location
ROOT_DIR = Path(__file__).resolve().parents[2]  # 2 levels up from implementations/

# Add the project root to sys.path
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Now you can import from configs/
from configs.config import MODEL_PATH, VIDEO_PATH
from src.Helpers.common_functions import detect_dominant_color_hsv
# from basicsr.archs.rrdbnet_arch import RRDBNet
# from realesrgan import RealESRGANer

yolo_model = YOLO(MODEL_PATH)

# Moving the model to GPU
yolo_model.eval().to('cuda')

# model_path = './models/RealESRGAN_x4plus.pth'

# model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)

# upsampler = RealESRGANer(
#     scale=4,
#     model_path=model_path,
#     model=model,
#     tile=0,
#     pre_pad=0,
#     half=False,  # ✅ Use half precision
#     device=torch.device('cuda')  # ✅ Use GPU
# )

# Initialize EasyOCR reader
reader = Reader(['en'], gpu=True)

# Function to detect color from BGR tuple using HSV color space
# Hard coded to red and blue for now, can be extended to other colors

# Open video
cap = cv2.VideoCapture(VIDEO_PATH)

# Configure video writer
fourcc = cv2.VideoWriter_fourcc(*'H264')

fps = cap.get(cv2.CAP_PROP_FPS)

frame_count = 0

# dynamically grab input’s FPS and frame size
fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# output_path = 'output_15sec_input_720p.mp4'

# out = cv2.VideoWriter(
#         output_path,
#         fourcc,
#         fps,
#         (width, height)
#     )

frame_count = 0
while cap.isOpened():
    crops = []
    ret, frame = cap.read()
    if not ret:
        break

    # Run detection
    results = yolo_model(frame)[0]
    frame_count += 1

    # Skipping to the part where model is in action
    if frame_count < 60:
        continue
    # Draw boxes
    for box in results.boxes:
        cls_id = int(box.cls[0])
        # Filter for just players
        if cls_id!=2:
            continue
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # bounding box

        # Create a crop of the detected object to upscale later
        crop = frame[y1:y2, x1:x2]

        # Get color name
        label = detect_dominant_color_hsv(crop)
        # crops.append(crop)
        conf = box.conf[0]

        upscale_method = 1

        # if upscale_method == 0:
        #     crop, _ = upsampler.enhance(crop, outscale=4)

        if upscale_method == 1:
            # Upscale by 2×
            scale = 10
            crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

        # Convert to grayscale
        # gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        # gray_img_e = clahe.apply(gray_img)

        # Create list of detected texts and confidences
        texts = []
        confs = []

        result_2 = reader.readtext(crop)
        # Print the results
        for detection in result_2:
            texts.append(detection[1])
            confs.append(detection[2])

        title = "YOLO Detection"
        if len(confs) != 0:
            id = confs.index(max(confs))
            text = texts[id]
            if text.isdigit():
                label += f" {text}"

        # Draw
        # cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f'{label}', (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow(title, frame)
    # Write processed frame to output video
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
