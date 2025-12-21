
import streamlit as st
import cv2
import numpy as np
from PIL import Image

COLOR_RANGES = {
    'Вода': {'color': (255, 0, 0), 'hsv': ([90, 30, 50], [130, 255, 255])},
    'Лес': {'color': (0, 255, 0), 'hsv': ([35, 40, 40], [85, 255, 255])},
    'Горы': {'color': (128, 128, 128), 'hsv': ([0, 0, 50], [30, 80, 150])},
    'Пустыня': {'color': (0, 165, 255), 'hsv': ([15, 30, 150], [35, 150, 255])},
    'Снег': {'color': (255, 255, 255), 'hsv': ([0, 0, 200], [180, 50, 255])},
}

def process_zones(image_bgr, selected_zones, color_ranges):
    """Обрабатывает изображение для нескольких зон"""
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    color_mask = np.zeros_like(image_bgr)
    contours_image = image_bgr.copy()
    results = []
    
    for zone in selected_zones:
        if zone in color_ranges:
            lower = np.array(color_ranges[zone]['hsv'][0])
            upper = np.array(color_ranges[zone]['hsv'][1])
            color = color_ranges[zone]['color']
            
            mask = cv2.inRange(hsv, lower, upper)
            color_mask[mask > 0] = color
            
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(contours_image, contours, -1, color, 2)
            
            area = cv2.countNonZero(mask)
            results.append({'name': zone, 'area': area, 'color': color})
    
    return color_mask, contours_image, results
st.set_page_config(page_title="Анализатор зон", layout="wide")
st.title("🌍 Анализатор географических зон")

uploaded_file = st.file_uploader("📁 Загрузите изображение", ['jpg', 'png'])

if uploaded_file:
    image = Image.open(uploaded_file)
    image_rgb = np.array(image)
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    zones = list(COLOR_RANGES.keys())
    selected = st.multiselect("🎯 Выберите зоны", zones, default=zones[:2])
    pixel_size = st.slider("📏 1 пиксель = (м²)", 0.1, 100.0, 1.0, 0.1)
    
    if selected:
        color_mask, contours_img, results = process_zones(image_bgr, selected, COLOR_RANGES)
        col1, col2, col3 = st.columns(3)
        with col1: st.image(image_rgb, caption="Исходное", use_column_width=True)
        with col2: st.image(color_mask, caption="Выделение", use_column_width=True)
        with col3: st.image(contours_img, caption="Контуры", use_column_width=True)
        st.subheader("📊 Результаты")
        for result in results:
            area_m2 = result['area'] * pixel_size
            color = result['color'][::-1]
            st.markdown(f"""
                <div style="display: flex; align-items: center; gap: 10px; margin: 5px 0;">
                    <div style="width: 20px; height: 20px; background-color: rgb{color}; 
                             border-radius: 3px;"></div>
                    <span><b>{result['name']}</b>: {result['area']:,} px ({area_m2:,.1f} м²)</span>
                </div>
            """, unsafe_allow_html=True)
    else:
        st.warning("⚠️ Выберите хотя бы одну зону")
else:
    st.info("👆 Загрузите изображение для анализа")
st.caption("🛰️ Анализатор зон")