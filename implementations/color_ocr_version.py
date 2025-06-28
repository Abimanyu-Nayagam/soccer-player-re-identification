import torch
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import cv2
from ultralytics import YOLO
from easyocr import Reader
# from basicsr.archs.rrdbnet_arch import RRDBNet
# from realesrgan import RealESRGANer

yolo_model = YOLO('./models/best.pt')

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
video_path = './Assignment Materials/Assignment Materials/15sec_input_720p.mp4'

# Function to detect color from BGR tuple using HSV color space
# Hard coded to red and blue for now, can be extended to other colors
def detect_dominant_color_hsv(crop):
    hsv_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    # Relaxed (but still robust) HSV ranges
    lower_red1 = np.array([0, 100, 60])
    upper_red1 = np.array([12, 255, 255])
    lower_red2 = np.array([165, 100, 60])
    upper_red2 = np.array([180, 255, 255])

     # Expanded blue range with relaxed S and V
    lower_blue = np.array([85, 50, 50])
    upper_blue = np.array([140, 255, 255])

    # Create masks
    mask_red1 = cv2.inRange(hsv_crop, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv_crop, lower_red2, upper_red2)
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)
    mask_blue = cv2.inRange(hsv_crop, lower_blue, upper_blue)

    # Count pixels
    red_count = cv2.countNonZero(mask_red)
    blue_count = cv2.countNonZero(mask_blue)
    total_pixels = crop.shape[0] * crop.shape[1]

    # Calculate color dominance ratios
    red_ratio = red_count / total_pixels
    blue_ratio = blue_count / total_pixels

    # Slightly lower threshold (e.g., 7%)
    if red_ratio > blue_ratio and red_ratio > 0.07:
        return "red"
    elif blue_ratio > red_ratio and blue_ratio > 0.07:
        return "blue"
    else:
        return "unknown"


# Open video
cap = cv2.VideoCapture(video_path)  # or 0 for webcam

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

    # Skipping to the part where model works as expected, to showcase the idea
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
