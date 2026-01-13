import os
import time
import logging
from typing import Optional

import streamlit as st
import numpy as np
from PIL import Image
import cv2

from config import (
    DEFAULT_MODEL, AVAILABLE_MODELS, DEFAULT_CONF, DEFAULT_IOU,
    DEFAULT_IMGSZ, DEFAULT_MAX_DET, DEFAULT_DEVICE, DEFAULT_LINE_THICKNESS,
    DEFAULT_SHOW_LABELS, DEFAULT_SHOW_CONF, DEFAULT_CAMERA_INDEX,
    ALLOWED_IMAGE_EXTENSIONS, ALLOWED_VIDEO_EXTENSIONS, TEMP_DIR
)
from inference import YOLOInference

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@st.cache_resource
def load_model(model_name: str, device: str) -> YOLOInference:
    """Загрузка и кеширование модели YOLO."""
    try:
        return YOLOInference(model_name, device)
    except Exception as e:
        st.error(f"Не удалось загрузить модель: {e}")
        logger.error(f"Ошибка загрузки модели: {e}")
        raise


def save_uploaded_file(uploaded_file) -> Optional[str]:
    """Сохранение загруженного файла во временную папку."""
    try:
        os.makedirs(TEMP_DIR, exist_ok=True)
        temp_path = os.path.join(TEMP_DIR, uploaded_file.name)
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        return temp_path
    except Exception as e:
        logger.error(f"Ошибка сохранения файла: {e}")
        st.error(f"Не удалось сохранить файл: {e}")
        return None


def cleanup_temp_files():
    """Очистка временного файла."""
    try:
        if os.path.exists(TEMP_DIR):
            for file in os.listdir(TEMP_DIR):
                file_path = os.path.join(TEMP_DIR, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)
    except Exception as e:
        logger.error(f"Ошибка при удалении временного файла: {e}")


def process_image_mode(model: YOLOInference, params: dict):
    """Процесс инференса на изображениях."""
    uploaded = st.file_uploader(
        "Загрузка изображения",
        type=ALLOWED_IMAGE_EXTENSIONS
    )

    if uploaded is not None:
        try:
            pil_img = Image.open(uploaded)
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Исходное изображение")
                st.image(pil_img, use_container_width=True)

            with col2:
                st.subheader("Результаты детекции")
                with st.spinner("Запуск инференса..."):
                    start_time = time.time()
                    out_img, results, stats = model.infer_image(
                        pil_img,
                        conf=params["conf"],
                        iou=params["iou"],
                        imgsz=params["imgsz"],
                        max_det=params["max_det"],
                        show_labels=params["show_labels"],
                        show_conf=params["show_conf"],
                        line_thickness=params["line_thickness"],
                        classes=params["classes"]
                    )
                    inference_time = time.time() - start_time

                st.image(out_img, use_container_width=True)

                st.metric("Время инференса", f"{inference_time:.3f}s")

                if stats:
                    st.write("**Детекции:**")
                    for class_name, count in stats.items():
                        st.write(f"- {class_name}: {count}")

                buf = io.BytesIO()
                out_img.save(buf, format="PNG")
                st.download_button(
                    "Скачать результаты",
                    data=buf.getvalue(),
                    file_name="detection_result.png",
                    mime="image/png"
                )

        except Exception as e:
            st.error(f"Ошибка обработки изображения: {e}")
            logger.error(f"Ошибка обработки изображения: {e}")


def process_video_mode(model: YOLOInference, params: dict):
    """Pежим инференса видеофайлов."""
    uploaded = st.file_uploader(
        "Загрузка видео",
        type=ALLOWED_VIDEO_EXTENSIONS
    )

    if uploaded is not None:
        temp_path = save_uploaded_file(uploaded)
        if temp_path is None:
            return

        try:
            cap = cv2.VideoCapture(temp_path)
            if not cap.isOpened():
                st.error("Не удалось открыть видеофайл")
                return

            frame_placeholder = st.empty()
            stats_placeholder = st.empty()
            stop_button = st.button("Остановка обработки")

            frame_count = 0
            total_time = 0

            while cap.isOpened() and not stop_button:
                ret, frame = cap.read()
                if not ret:
                    break

                start_time = time.time()
                results = model.predict(
                    frame,
                    conf=params["conf"],
                    iou=params["iou"],
                    imgsz=params["imgsz"],
                    max_det=params["max_det"],
                    classes=params["classes"]
                )

                from utils import draw_results_on_frame, get_detection_stats
                frame = draw_results_on_frame(
                    frame, results,
                    params["show_labels"],
                    params["show_conf"],
                    params["line_thickness"]
                )

                inference_time = time.time() - start_time
                total_time += inference_time
                frame_count += 1

                fps = 1.0 / inference_time if inference_time > 0 else 0
                avg_fps = frame_count / total_time if total_time > 0 else 0

                cv2.putText(
                    frame, f"FPS: {fps:.1f} | Avg: {avg_fps:.1f}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
                )

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

                stats = get_detection_stats(results)
                if stats:
                    stats_text = " | ".join([f"{k}: {v}" for k, v in stats.items()])
                    stats_placeholder.text(f"Frame {frame_count}: {stats_text}")

            cap.release()
            st.success(f"Обработано {frame_count} кадров. Средний FPS: {avg_fps:.2f}")

        except Exception as e:
            st.error(f"Ошибка обработки видео: {e}")
            logger.error(f"Ошибка обработки видео: {e}")

        finally:
            cleanup_temp_files()


def process_webcam_mode(model: YOLOInference, params: dict, camera_index: int):
    """Процесс инференса через веб-камеру."""
    if st.button("Запуск веб-камеры"):
        try:
            cap = cv2.VideoCapture(camera_index)
            if not cap.isOpened():
                st.error("Не удалось открыть веб-камеру. Проверьте индекс камеры и права доступа.")
                return

            frame_placeholder = st.empty()
            stats_placeholder = st.empty()
            stop_button = st.button("Остановить веб-камеру")

            frame_count = 0
            total_time = 0

            while cap.isOpened() and not stop_button:
                ret, frame = cap.read()
                if not ret:
                    break

                start_time = time.time()
                results = model.predict(
                    frame,
                    conf=params["conf"],
                    iou=params["iou"],
                    imgsz=params["imgsz"],
                    max_det=params["max_det"],
                    classes=params["classes"]
                )

                from utils import draw_results_on_frame, get_detection_stats
                frame = draw_results_on_frame(
                    frame, results,
                    params["show_labels"],
                    params["show_conf"],
                    params["line_thickness"]
                )

                inference_time = time.time() - start_time
                total_time += inference_time
                frame_count += 1

                fps = 1.0 / inference_time if inference_time > 0 else 0

                cv2.putText(
                    frame, f"FPS: {fps:.1f}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
                )

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

                stats = get_detection_stats(results)
                if stats:
                    stats_text = " | ".join([f"{k}: {v}" for k, v in stats.items()])
                    stats_placeholder.text(stats_text)

            cap.release()

        except Exception as e:
            st.error(f"Ошибка веб-камеры: {e}")
            logger.error(f"Ошибка веб-камеры: {e}")


def main():
    """Основное приложение."""
    st.set_page_config(
        page_title="Детекция объектов через YOLOv8",
        page_icon="🎯",
        layout="wide"
    )

    st.title("Детекция объектов через YOLOv8")
    st.caption("Детекция объектов в реальном времени с помощью YOLOv8 и Streamlit")

    with st.sidebar:
        st.header("Конфигурация")

        model_name = st.selectbox("Model", AVAILABLE_MODELS, index=2)
        device = st.selectbox("Device", ["cpu", "cuda"], index=0)

        st.divider()

        st.subheader("Inference Parameters")
        conf = st.slider("Confidence Threshold", 0.0, 1.0, DEFAULT_CONF, 0.01)
        iou = st.slider("IoU (NMS)", 0.0, 1.0, DEFAULT_IOU, 0.01)
        imgsz = st.select_slider("Image Size", options=[320, 416, 512, 640, 768, 960], value=DEFAULT_IMGSZ)
        max_det = st.number_input("Max Detections", min_value=1, max_value=300, value=DEFAULT_MAX_DET, step=1)

        st.divider()

        st.subheader("Visualization")
        show_labels = st.checkbox("Show Labels", DEFAULT_SHOW_LABELS)
        show_conf = st.checkbox("Show Confidence", DEFAULT_SHOW_CONF)
        line_thickness = st.slider("Line Thickness", 1, 8, DEFAULT_LINE_THICKNESS)

        st.divider()

        mode = st.radio("Mode", ["Image", "Video", "Webcam"])

        if mode == "Webcam":
            camera_index = st.number_input("Camera Index", value=DEFAULT_CAMERA_INDEX, step=1)

    try:
        model = load_model(model_name, device)

        class_names = model.get_class_names()
        selected_classes = None

        if class_names:
            with st.sidebar:
                st.divider()
                st.subheader("Class Filter")
                use_filter = st.checkbox("Enable class filtering")
                if use_filter:
                    selected_class_names = st.multiselect(
                        "Select classes to detect",
                        options=list(class_names.values())
                    )
                    if selected_class_names:
                        selected_classes = [
                            k for k, v in class_names.items() if v in selected_class_names
                        ]

        params = {
            "conf": conf,
            "iou": iou,
            "imgsz": imgsz,
            "max_det": max_det,
            "show_labels": show_labels,
            "show_conf": show_conf,
            "line_thickness": line_thickness,
            "classes": selected_classes
        }

        if mode == "Image":
            process_image_mode(model, params)
        elif mode == "Video":
            process_video_mode(model, params)
        elif mode == "Webcam":
            process_webcam_mode(model, params, camera_index)

    except Exception as e:
        st.error(f"Application error: {e}")
        logger.error(f"Application error: {e}")


if __name__ == "__main__":
    import io
    main()
