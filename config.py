from typing import Dict

# Конфигурация модели
DEFAULT_MODEL = "yolov8m.pt"
AVAILABLE_MODELS = ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt"]

# Значения по умолчанию для инференса
DEFAULT_CONF = 0.25
DEFAULT_IOU = 0.45
DEFAULT_IMGSZ = 640
DEFAULT_MAX_DET = 100
DEFAULT_DEVICE = "cpu"

# Значения по умолчанию для пользовательского интерфейса
DEFAULT_LINE_THICKNESS = 2
DEFAULT_SHOW_LABELS = True
DEFAULT_SHOW_CONF = True

# Настройки камеры
DEFAULT_CAMERA_INDEX = 0

# Настройки файлов
ALLOWED_IMAGE_EXTENSIONS = ["jpg", "jpeg", "png", "bmp", "webp"]
ALLOWED_VIDEO_EXTENSIONS = ["mp4", "avi", "mov", "mkv"]
TEMP_DIR = "temp"

# Цвета баундинг боксов (BGR формат)
BOX_COLORS: Dict[str, tuple] = {
    "default": (0, 255, 0),
    "person": (255, 0, 0),
    "car": (0, 0, 255),
}