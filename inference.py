import cv2
import numpy as np
from PIL import Image
from typing import List, Optional, Tuple
from ultralytics import YOLO
from ultralytics.engine.results import Results
import logging

from utils import draw_results_on_frame, get_detection_stats

logger = logging.getLogger(__name__)


class YOLOInference:
    """Обработка инференса модели YOLO."""

    def __init__(self, model_path: str, device: str = "cpu"):
        """Инициализация инференса YOLO."""
        self.model_path = model_path
        self.device = device
        self.model: Optional[YOLO] = None
        self._load_model()

    def _load_model(self) -> None:
        """Загрузка модели YOLO."""
        try:
            self.model = YOLO(self.model_path)
            logger.info(f"Модель {self.model_path} успешно загружена на {self.device}")
        except Exception as e:
            logger.error(f"Не удалось загрузить модель {self.model_path}: {e}")
            raise

    def predict(
        self,
        source: np.ndarray,
        conf: float = 0.25,
        iou: float = 0.45,
        imgsz: int = 640,
        max_det: int = 100,
        classes: Optional[List[int]] = None
    ) -> List[Results]:
        """Запуск инференса на изображениях."""
        try:
            results = self.model.predict(
                source=source,
                conf=conf,
                iou=iou,
                imgsz=imgsz,
                max_det=max_det,
                device=self.device,
                classes=classes,
                verbose=False
            )
            return results
        except Exception as e:
            logger.error(f"Инференс не удался: {e}")
            return []

    def infer_image(
        self,
        pil_img: Image.Image,
        conf: float = 0.25,
        iou: float = 0.45,
        imgsz: int = 640,
        max_det: int = 100,
        show_labels: bool = True,
        show_conf: bool = True,
        line_thickness: int = 2,
        classes: Optional[List[int]] = None
    ) -> Tuple[Image.Image, List[Results], dict]:
        """Inference on single image with annotation."""
        try:
            img = np.array(pil_img.convert("RGB"))
            results = self.predict(img, conf, iou, imgsz, max_det, classes)

            frame_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            frame_bgr = draw_results_on_frame(
                frame_bgr, results, show_labels, show_conf, line_thickness
            )
            out_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            stats = get_detection_stats(results)

            return Image.fromarray(out_rgb), results, stats

        except Exception as e:
            logger.error(f"Инференс изображения не удался: {e}")
            return pil_img, [], {}

    def get_class_names(self) -> dict:
        """Получение имен классов."""
        if self.model is not None:
            return self.model.names
        return {}
