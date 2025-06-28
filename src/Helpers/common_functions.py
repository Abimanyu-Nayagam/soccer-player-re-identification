import numpy as np
import cv2

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