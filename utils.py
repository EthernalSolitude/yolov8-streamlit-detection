import cv2
import numpy as np
from typing import List, Optional, Dict, Tuple
from ultralytics.engine.results import Results
import logging

logger = logging.getLogger(__name__)


def draw_results_on_frame(
    frame_bgr: np.ndarray,
    results: List[Results],
    show_labels: bool = True,
    show_conf: bool = True,
    line_thickness: int = 2,
    box_colors: Optional[Dict[str, tuple]] = None
) -> np.ndarray:
    """Отображение результатов детекции на кадре."""
    if box_colors is None:
        box_colors = {"default": (0, 255, 0)}

    try:
        for r in results:
            if not hasattr(r, "boxes") or r.boxes is None:
                continue

            names = getattr(r, "names", None)
            xyxy = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy() if r.boxes.conf is not None else None
            clss = r.boxes.cls.cpu().numpy() if r.boxes.cls is not None else None

            for i, box in enumerate(xyxy):
                x1, y1, x2, y2 = box[:4].astype(int)
                conf_val = float(confs[i]) if confs is not None else None
                cls_id = int(clss[i]) if clss is not None else None

                class_name = None
                if cls_id is not None and names is not None:
                    class_name = names.get(cls_id, str(cls_id))

                color = box_colors.get(class_name, box_colors.get("default", (0, 255, 0)))

                label = None
                if show_labels and class_name:
                    label = class_name
                    if show_conf and conf_val is not None:
                        label = f"{label} {conf_val:.2f}"
                elif show_conf and conf_val is not None:
                    label = f"{conf_val:.2f}"

                cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, line_thickness)

                if label:
                    ((tw, th), _) = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    th_box = th + 6
                    cv2.rectangle(frame_bgr, (x1, y1 - th_box), (x1 + tw + 4, y1), color, -1)
                    cv2.putText(
                        frame_bgr, label, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA
                    )

    except Exception as e:
        logger.error(f"Ошибка при отрисовки результатов: {e}")

    return frame_bgr


def calculate_fps(start_time: float, end_time: float) -> float:
    """Рассчет количества кадров в секунду."""
    elapsed = end_time - start_time
    return 1.0 / elapsed if elapsed > 0 else 0.0


def get_detection_stats(results: List[Results]) -> Dict[str, int]:
    """Получение результатов детекции."""
    stats = {}

    try:
        for r in results:
            if not hasattr(r, "boxes") or r.boxes is None:
                continue

            names = getattr(r, "names", None)
            clss = r.boxes.cls.cpu().numpy() if r.boxes.cls is not None else None

            if clss is not None and names is not None:
                for cls_id in clss:
                    class_name = names.get(int(cls_id), "unknown")
                    stats[class_name] = stats.get(class_name, 0) + 1

    except Exception as e:
        logger.error(f"Ошибка при вычислении результатов детекции: {e}")

    return stats
