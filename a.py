import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from utils.color_ranges import COLOR_RANGES
from utils.process import process_zones
from utils.db import (init_db, register_user, authenticate_user, 
                      save_analysis_to_disk, get_user_snapshots, 
                      get_snapshot_full_info, get_unique_territories, 
                      get_territory_history, get_snapshots_for_territory,
                      create_session, get_user_by_token, delete_session, get_username_by_id)

init_db()

st.set_page_config(page_title="Анализатор зон", layout="wide")

if 'user_id' not in st.session_state:
    st.session_state['user_id'] = None
    st.session_state['username'] = None

token = st.query_params.get("token")
if token and st.session_state['user_id'] is None:
    uid = get_user_by_token(token)
    if uid:
        st.session_state['user_id'] = uid
        st.session_state['username'] = get_username_by_id(uid)
    else:
        st.query_params.clear()

def get_contours_for_zone(hsv1, hsv2, hsv_values):
    try:
        lower = np.array(hsv_values[:3], dtype="uint8")
        upper = np.array(hsv_values[3:], dtype="uint8")
    except:
        return None, None

    mask1 = cv2.inRange(hsv1, lower, upper)
    mask2 = cv2.inRange(hsv2, lower, upper)

    kernel = np.ones((5, 5), np.uint8)
    mask1 = cv2.morphologyEx(mask1, cv2.MORPH_OPEN, kernel)
    mask2 = cv2.morphologyEx(mask2, cv2.MORPH_OPEN, kernel)

    diff_loss = cv2.subtract(mask1, mask2)
    diff_gain = cv2.subtract(mask2, mask1)

    cnt_loss, _ = cv2.findContours(diff_loss, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt_gain, _ = cv2.findContours(diff_gain, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not cnt_loss and not cnt_gain:
        return None, None
        
    return cnt_loss, cnt_gain

def process_visual_comparison(img1_path, img2_path, biom, zone_name):
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    
    if img1 is None or img2 is None:
        return None, "Не удалось загрузить изображения"

    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

    hsv1 = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
    hsv2 = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)

    try:
        hsv_values = COLOR_RANGES[biom][zone_name]['hsv']
        lower = np.array(hsv_values[:3], dtype="uint8")
        upper = np.array(hsv_values[3:], dtype="uint8")
        
    except KeyError:
        return None, f"Ошибка данных: зона '{zone_name}' не найдена"
    except Exception as e:
        return None, f"Ошибка обработки: {str(e)}"

    mask1 = cv2.inRange(hsv1, lower, upper)
    mask2 = cv2.inRange(hsv2, lower, upper)

    kernel = np.ones((5, 5), np.uint8)
    mask1 = cv2.morphologyEx(mask1, cv2.MORPH_OPEN, kernel)
    mask2 = cv2.morphologyEx(mask2, cv2.MORPH_OPEN, kernel)

    diff_loss = cv2.subtract(mask1, mask2)
    diff_gain = cv2.subtract(mask2, mask1)

    result_img = img2.copy()
    
    contours_loss, _ = cv2.findContours(diff_loss, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(result_img, contours_loss, -1, (0, 0, 255), 2)  

    contours_gain, _ = cv2.findContours(diff_gain, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(result_img, contours_gain, -1, (0, 255, 0), 2)  

    return cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB), "OK"

def show_login_page():
    st.title('Войдите или зарегистрируйтесь')
    t1, t2 = st.tabs(['Войти', 'Регистрация'])
    
    with t1: 
        u = st.text_input('Логин', key='l_u')
        p = st.text_input('Пароль', key='l_p', type="password")
        if st.button('Войти'):
            uid = authenticate_user(u, p)
            if uid:
                token = create_session(uid)
                st.query_params["token"] = token
                st.session_state['user_id'] = uid
                st.session_state['username'] = u
                st.rerun()
            else:
                st.error("Неверный логин или пароль")
    
    with t2:
        reg_u = st.text_input('Придумайте логин', key='r_u')
        reg_p = st.text_input('Придумайте пароль', key='r_p', type="password")
        if st.button('Зарегистрироваться'):
            if register_user(reg_u, reg_p):
                st.success("Вы зарегистрированы. Теперь войдите.")
            else:
                st.error("Пользователь с таким именем уже существует.")

def show_new_analysis():
    st.header("Новый анализ")
    territory_name = st.text_input("Название территории", "Моя территория")

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
                st.image(image_rgb, caption="Исходное", use_column_width=True)
            with col2:
                st.image(color_mask, caption="Выделение", use_column_width=True)
            with col3:
                st.image(contours_img, caption="Контуры", use_column_width=True)
            
            st.subheader("Результаты")
            
            measurements_to_save = []

            for result in results:
                area_m2 = result['area_pixels'] * pixel_size
                color = result['color'][::-1]
                
                measurements_to_save.append({
                    'category': biom,
                    'subzone': result['subzone'],
                    'area_pixels': result['area_pixels'],
                    'area_m2': area_m2
                })

                st.markdown(f"""
                    <div style="display: flex; align-items: center; gap: 10px; margin: 5px 0;">
                        <div style="width: 20px; height: 20px; background-color: rgb{color};
                                border-radius: 3px;"></div>
                        <span><b>{result['subzone']}</b>: {result['area_pixels']:,} px ({area_m2:,.1f} м²)</span>
                    </div>
                """, unsafe_allow_html=True)
            
            st.markdown("---")
            if st.button("Сохранить в Архив"):
                contours_bgr = cv2.cvtColor(contours_img, cv2.COLOR_RGB2BGR)
                
                success, msg = save_analysis_to_disk(
                    user_id=st.session_state['user_id'],
                    img_orig_bgr=image_bgr,
                    img_proc_bgr=contours_bgr,
                    filename_raw=uploaded_file.name,
                    territory=territory_name,
                    measurements=measurements_to_save
                )
                
                if success:
                    st.success(msg)
                else:
                    st.error(msg)
        else:
            st.warning("Выберите хотя бы одну зону")
    else:
        st.info("Загрузите изображение для анализа")

def show_archive():
    st.header("Архив снимков")
    snapshots = get_user_snapshots(st.session_state['user_id'])
    
    if not snapshots:
        st.info("В архиве пока пусто")
        return

    options = {f"{row[1]} (от {row[2]})": row[0] for row in snapshots}
    selected_label = st.selectbox("Выберите сохраненный анализ:", list(options.keys()))
    
    if selected_label:
        snapshot_id = options[selected_label]
        data = get_snapshot_full_info(snapshot_id)
        
        if data['error']:
            st.error(data['error'])
            return

        st.caption(f"Дата загрузки: {data['date']}")
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(data['image_orig'], caption="Оригинал", use_column_width=True)
        with col2:
            st.image(data['image_proc'], caption="Обработка", use_column_width=True)
            
        st.subheader("Сохраненные данные")
        for m in data['measurements']:
            st.write(f"- **{m['subzone']}**: {m['area_pixels']} px / {m['area_m2']:.1f} м²")

def show_timeline():
    st.header("Динамика изменений (График)")
    
    territories = get_unique_territories(st.session_state['user_id'])
    
    if not territories:
        st.info("У вас нет сохраненных данных.")
        return

    selected_terr = st.selectbox("Выберите территорию для анализа:", territories)
    
    history_data = get_territory_history(st.session_state['user_id'], selected_terr)
    
    if not history_data:
        st.warning("Нет данных для отображения.")
        return

    df = pd.DataFrame(history_data, columns=['Date', 'Zone', 'Area_m2'])
    df['Date'] = pd.to_datetime(df['Date'])

    st.subheader(f"Изменение площади: {selected_terr}")
    
    chart_data = df.pivot_table(index='Date', columns='Zone', values='Area_m2', aggfunc='sum')
    st.line_chart(chart_data)

    st.subheader("Аналитический отчет")
    
    zones = df['Zone'].unique()
    
    if len(chart_data) < 2:
        st.info("Нужно минимум 2 снимка для анализа.")
    else:
        for zone in zones:
            if zone in chart_data.columns:
                series = chart_data[zone].dropna()
                
                if len(series) >= 2:
                    start_val = series.iloc[0]
                    end_val = series.iloc[-1]
                    
                    diff = end_val - start_val
                    percent = (diff / start_val * 100) if start_val != 0 else 0
                    
                    if diff > 0:
                        status = "УВЕЛИЧИЛАСЬ"
                        delta_color = "normal"
                    elif diff < 0:
                        status = "УМЕНЬШИЛАСЬ"
                        delta_color = "inverse"
                    else:
                        status = "НЕ ИЗМЕНИЛАСЬ"
                        delta_color = "off"

                    st.metric(
                        label=f"Зона: {zone}",
                        value=f"{end_val:.1f} м²",
                        delta=f"{diff:+.1f} м² ({percent:+.1f}%)",
                        delta_color=delta_color
                    )
                    st.divider()

def show_visual_comparison():
    st.header("Автоматический поиск изменений")
    
    territories = get_unique_territories(st.session_state['user_id'])
    
    if not territories:
        st.info("Нет сохраненных территорий.")
        return
        
    selected_terr = st.selectbox("Выберите территорию:", territories, key="viz_terr")
    
    snaps = get_snapshots_for_territory(st.session_state['user_id'], selected_terr)
    
    if len(snaps) < 2:
        st.warning("Для сравнения нужно минимум 2 снимка.")
        return
        
    snap_options = {f"{s[1]} (ID: {s[0]})": s[2] for s in snaps}
    labels = list(snap_options.keys())
    
    c1, c2 = st.columns(2)
    with c1:
        label1 = st.selectbox("Снимок 1 (Старый):", labels, index=0)
    with c2:
        label2 = st.selectbox("Снимок 2 (Новый):", labels, index=len(labels)-1)
        
    path1 = snap_options[label1]
    path2 = snap_options[label2]
    
    if st.button("Найти изменения"):
        img1 = cv2.imread(path1)
        img2 = cv2.imread(path2)
        
        if img1 is None or img2 is None:
            st.error("Ошибка загрузки изображений с диска")
            return

        if img1.shape != img2.shape:
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
            
        hsv1 = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
        hsv2 = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)
        
        st.info("Сканирование всех зон на наличие изменений...")
        
        detected_changes = {}
        
        for biome_name, zones_dict in COLOR_RANGES.items():
            for zone_name, props in zones_dict.items():
                loss, gain = get_contours_for_zone(hsv1, hsv2, props['hsv'])
                if loss or gain:
                    full_name = f"{biome_name}: {zone_name}"
                    detected_changes[full_name] = {'loss': loss, 'gain': gain}
        
        if not detected_changes:
            st.warning("Значимых изменений в известных зонах не найдено.")
            st.image(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB), caption="Снимок 1", width=350)
            st.image(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB), caption="Снимок 2", width=350)
            return
            
        st.session_state['viz_changes'] = detected_changes
        st.session_state['viz_img1'] = img1
        st.session_state['viz_img2'] = img2
        
    if 'viz_changes' in st.session_state:
        changes = st.session_state['viz_changes']
        img1 = st.session_state['viz_img1']
        img2 = st.session_state['viz_img2']
        
        all_zones = list(changes.keys())
        selected_zones = st.multiselect(
            "Обнаружены изменения в зонах (удалите лишние):", 
            all_zones, 
            default=all_zones
        )
        
        result_img = img2.copy()
        
        overlay = result_img.copy()
        cv2.addWeighted(overlay, 0.7, result_img, 0.3, 0, result_img)
        
        for zone_key in selected_zones:
            data = changes[zone_key]
            if data['loss']:
                cv2.drawContours(result_img, data['loss'], -1, (0, 0, 255), 2)
            if data['gain']:
                cv2.drawContours(result_img, data['gain'], -1, (0, 255, 0), 2)
                
        col_old, col_new = st.columns(2)
        
        with col_old:
            st.subheader("Было (Старый снимок)")
            st.image(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB), use_column_width=True)
            
        with col_new:
            st.subheader("Стало (С отклонениями)")
            st.image(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB), use_column_width=True)
            
        st.success("Легенда: 🔴 Красный = Исчезло, 🟢 Зеленый = Появилось")

def show_app():
    st.sidebar.title("Меню")
    page = st.sidebar.radio("Перейти:", ["Новый анализ", "Архив", "Динамика (График)", "Сравнение (Фото)"])
    
    st.sidebar.divider()
    st.sidebar.write(f"Логин: {st.session_state['username']}")
    if st.sidebar.button("Выйти"):
        token = st.query_params.get("token")
        if token:
            delete_session(token)
        st.query_params.clear()
        st.session_state['user_id'] = None
        st.session_state['username'] = None
        st.rerun()

    if page == "Новый анализ":
        show_new_analysis()
    elif page == "Архив":
        show_archive()
    elif page == "Динамика (График)":
        show_timeline()
    elif page == "Сравнение (Фото)":
        show_visual_comparison()

if st.session_state['user_id'] is None:
    show_login_page()
else:
    show_app()