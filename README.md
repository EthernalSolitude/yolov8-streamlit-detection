# YOLOv8 Object Detection - Streamlit Application

Веб-приложение для детекции объектов в реальном времени с использованием YOLOv8 и Streamlit. Поддерживает обработку изображений, видеофайлов и веб-камеры.

## Описание

Это приложение позволяет запускать модели YOLOv8 для детекции объектов через удобный веб-интерфейс. Вы можете настраивать параметры детекции, фильтровать классы объектов и получать статистику в реальном времени.

Основные возможности:
- Детекция объектов на изображениях с возможностью скачать результат
- Обработка видеофайлов с отображением FPS
- Работа с веб-камерой в реальном времени
- Настройка параметров: confidence threshold, IoU, размер изображения
- Фильтрация по классам объектов (80 классов COCO dataset)
- Выбор между моделями разного размера: от быстрых до точных
- Поддержка CPU и CUDA для ускорения на GPU
- Docker-контейнеризация с оптимизированным образом для запуска на CPU

## Установка и запуск

### Вариант 1: Локальный запуск (с поддержкой GPU)

### Клонирование репозитория

```bash
git clone https://github.com/EthernalSolitude/yolov8-streamlit-detection.git
cd yolov8-streamlit-detection
```

### Установка зависимостей

Рекомендуется использовать виртуальное окружение (venv или conda).

```bash
pip install -r requirements.txt
```

При первом запуске автоматически загружается нужная модель.

### Запуск приложения

```bash
streamlit run app.py
```

Приложение откроется в браузере по адресу http://localhost:8501

### Вариант 2: Docker

Docker-образ использует CPU-оптимизированную версию PyTorch для портативности и уменьшенного размера.
GPU-ускорение доступно только при локальном запуске с установленной CUDA-версией PyTorch.

### Сборка образа

```bash
git clone https://github.com/EthernalSolitude/yolov8-streamlit-detection.git
cd yolov8-streamlit-detection
docker build -t yolo-app .
```

### Запуск контейнера

```bash
docker run -d --name yolo-app -p 8501:8501 -v yolo-cache:/root/.cache/ultralytics --restart unless-stopped yolo-app
```
Приложение будет доступно по адресу: http://localhost:8501

## Использование

### Работа с изображениями

1. Выберите режим "Image" в боковой панели
2. Настройте параметры детекции (можно оставить значения по умолчанию)
3. Загрузите изображение через интерфейс
4. Посмотрите результаты и скачайте аннотированное изображение

### Работа с видео

1. Выберите режим "Video"
2. Загрузите видеофайл
3. Наблюдайте за обработкой кадров и статистикой
4. Остановите обработку кнопкой "Остановка обработки"

### Работа с веб-камерой

1. Выберите режим "Webcam"
2. Укажите индекс камеры (0 для основной камеры)
3. Нажмите "Запуск веб-камеры"
4. Остановите через кнопку "Остановить веб-камеру"

## Настройка параметров

**Выбор модели:**
- YOLOv8n - самая быстрая, подходит для CPU
- YOLOv8s - быстрая с хорошей точностью
- YOLOv8m - баланс скорости и точности (рекомендуется)
- YOLOv8l - высокая точность, требуется GPU
- YOLOv8x - максимальная точность, требуется GPU

**Основные параметры:**
- Confidence Threshold - минимальная уверенность для детекции (по умолчанию 0.25)
- IoU - порог для устранения дублирующихся детекций (по умолчанию 0.45)
- Image Size - размер входного изображения, больше = точнее, но медленнее (по умолчанию 640x640)
- Max Detections - максимальное количество объектов на кадре (по умолчанию 100)

**Визуализация:**
- Show Labels - показывать названия классов
- Show Confidence - показывать уверенность детекции
- Line Thickness - толщина рамок вокруг объектов

**Фильтрация классов:**
Включите "Enable class filtering" чтобы выбрать детекцию только нужных классов объектов (доступны все классы из датасета COCO).

## Структура проекта

```
yolov8-streamlit-detection/
├── app.py                      
├── config.py                   
├── inference.py                
├── utils.py                   
├── requirements.txt            
├── requirements-docker.txt     
├── Dockerfile                  
├── .dockerignore               
└── README.md                              
```

**app.py** - главный файл с интерфейсом и логикой обработки разных режимов

**config.py** - хранит все настройки по умолчанию и списки поддерживаемых форматов

**inference.py** - обертка над YOLOv8 для удобной работы с моделью

**utils.py** - функции для отрисовки результатов и подсчета статистики

**requirements.txt** - полный набор зависимостей для локальной разработки (включая PyTorch с CUDA)

**requirements-docker.txt** - оптимизированные зависимости для Docker (PyTorch CPU-версия устанавливается отдельно в Dockerfile)

## Примеры кода

### Обработка изображений

```python
from inference import YOLOInference
from PIL import Image

# Загрузка модели
model = YOLOInference("yolov8m.pt", device="cpu")

# Детекция на изображении
image = Image.open("example.jpg")
result_image, results, stats = model.infer_image(image, conf=0.25)

# Сохранение результата
result_image.save("result.jpg")
print(f"Найдено объектов: {stats}")
```

### Обработка видео

```python
import cv2
from inference import YOLOInference

model = YOLOInference("yolov8n.pt")
cap = cv2.VideoCapture("video.mp4")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(frame, conf=0.3)
    # Обработка результатов...

cap.release()
```

### Детекция с фильтрацией классов

```python
from inference import YOLOInference
from PIL import Image

model = YOLOInference("yolov8s.pt")

# Детектировать только людей и машины (индексы 0 и 2)
image = Image.open("street.jpg")
result_image, results, stats = model.infer_image(
    image,
    conf=0.25,
    classes=[0, 2]
)

print(f"Обнаружено: {stats}")
```

