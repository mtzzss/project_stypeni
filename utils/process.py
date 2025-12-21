import cv2
import numpy as np


def process_zones(image_bgr, category, selected_subzones,
                  color_ranges):
    hsv_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    color_mask = np.zeros_like(image_bgr)
    contours_image = image_bgr.copy()

    results = []

    if category in color_ranges:

        for subzone in selected_subzones:
            if subzone in color_ranges[category]:
                subzone_info = color_ranges[category][subzone]

                hsv_range = subzone_info['hsv']
                lower = hsv_range[:3]
                upper = hsv_range[3:]
                mask = cv2.inRange(hsv_image, lower, upper)
                color = subzone_info['color']
                color_mask[mask > 0] = color
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(contours_image, contours, -1, color, 2)
                area_pixels = cv2.countNonZero(mask)
                results.append({
                    'category': category,
                    'subzone': subzone,
                    'area_pixels': area_pixels,
                    'color': color
                })

    return color_mask, contours_image, results
