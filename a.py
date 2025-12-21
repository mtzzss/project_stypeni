import streamlit as st
import cv2
import numpy as np
from PIL import Image
from utils.color_ranges import COLOR_RANGES
from utils.process import process_zones


st.set_page_config(page_title="Анализатор зон", layout="wide")
st.title("Анализатор географических зон")

uploaded_file = st.file_uploader("Загрузите изображение", ['jpg', 'png'])

if uploaded_file:
    image = Image.open(uploaded_file)
    image_rgb = np.array(image)
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    bioms = list(COLOR_RANGES.keys())
    biom = st.selectbox("Выберите категорию", bioms)
    zones = list(COLOR_RANGES[biom].keys())
    selected = st.multiselect("Выберите зоны", zones, default=zones[:2])
    pixel_size = st.slider(" 1 пиксель = (м²)", 0.1, 100.0, 1.0, 0.1)

    if selected:
        color_mask, contours_img, results = process_zones(
            image_bgr, biom, selected, COLOR_RANGES)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.image(image_rgb, caption="Исходное",
                     use_column_width=True)
        with col2:
            st.image(color_mask, caption="Выделение",
                     use_column_width=True)
        with col3:
            st.image(contours_img, caption="Контуры",
                     use_column_width=True)
        st.subheader("Результаты")
        for result in results:
            area_m2 = result['area_pixels'] * pixel_size
            color = result['color'][::-1]
            st.markdown(f"""
                <div style="display: flex; align-items: center; gap: 10px; margin: 5px 0;">
                    <div style="width: 20px; height: 20px; background-color: rgb{color};
                             border-radius: 3px;"></div>
                    <span><b>{result['subzone']}</b>: {result['area_pixels']:,} px ({area_m2:,.1f} м²)</span>
                </div>
            """, unsafe_allow_html=True)
    else:
        st.warning("Выберите хотя бы одну зону")
else:
    st.info("Загрузите изображение для анализа")
st.caption("Анализатор зон")