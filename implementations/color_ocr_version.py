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
def hsv_to_color_name_from_tuple(bgr_tuple):
    # Convert BGR tuple to a 1x1 image for OpenCV color conversion
    bgr_array = np.array([[bgr_tuple]], dtype=np.uint8)
    hsv = cv2.cvtColor(bgr_array, cv2.COLOR_BGR2HSV)[0][0]
    h, s, v = hsv

    if v < 50:
        return 'black'
    elif s < 30 and v > 200:
        return 'white'
    elif s < 40:
        return 'gray'
    elif 0 <= h <= 10 or h >= 160:
        return 'red'
    elif 11 <= h <= 25:
        return 'orange'
    elif 26 <= h <= 34:
        return 'yellow'
    elif 35 <= h <= 85:
        return 'green'
    elif 86 <= h <= 130:
        return 'blue'
    elif 131 <= h <= 159:
        return 'purple'
    else:
        return 'unknown'

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
    # if frame_count < 50:
    #     continue
    # Draw boxes
    for box in results.boxes:
        cls_id = int(box.cls[0])
        # Filter for just players
        if cls_id!=2:
            continue
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # bounding box

        # Create a crop of the detected object to upscale later
        crop = frame[y1:y2, x1:x2]

        # Processing for finding the most common color in the crop
        # Reshape image to a list of pixels
        pixels = crop.reshape(-1, 3)  # shape becomes (num_pixels, 3)

        # Convert each pixel to a tuple (hashable for counting)
        pixel_tuples = [tuple(pixel) for pixel in crop.reshape(-1, 3)]
        
        most_common_pixel = Counter(pixel_tuples).most_common(1)[0][0]

        # Get color name
        label = hsv_to_color_name_from_tuple(most_common_pixel)
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
