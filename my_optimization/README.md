TensorRT: Inference Optimization
Один Docker-контейнер для оптимизации инференса ONNX-моделей через TensorRT.

Обзор
#	Модель	Задача	Лицензия	Особенность
1	TensorRT 10.x	Inference optimization	NVIDIA SLA (бесплатно)	2-5x ускорение относительно ONNX Runtime
Что это даёт
TensorRT — компилятор NVIDIA для максимально быстрого инференса ONNX-моделей на GPU. Даёт 2-5x ускорение относительно ONNX Runtime за счёт:

Layer fusion — объединение слоёв в один kernel.

Kernel auto-tuning — подбор оптимальных kernel'ов под конкретную GPU.

FP16/INT8/FP8 квантизация — снижение точности для ускорения.

Dynamic shapes — поддержка переменных размеров входа.

Multi-head attention fusion — оптимизация трансформеров.

Применение: production deployment, latency-critical задачи, edge-устройства (Jetson), real-time inference.

Лицензия
NVIDIA SLA (Software License Agreement):

Проприетарная, но бесплатная для использования.

SDK лицензирован для разработки приложений только для систем с NVIDIA GPU.

Runtime files (.so, .dll) распространяемы.

Можно использовать в коммерческих продуктах без раскрытия кода.

Совместимость версий
Ключевое правило: версия TensorRT на машине сборки engine должна совпадать с версией на машине инференса. Иначе engine не загрузится.

Актуальные версии (2026):

Container	CUDA Toolkit	TensorRT
25.06	12.9.1	10.11.0
25.08	13.0	10.13.2
Базовый образ: nvcr.io/nvidia/tensorrt:25.06-py3 — CUDA 12.9.1, TensorRT 10.11.0.

Требования:

CUDA 11.0+.

PyTorch 2.0+ (для загрузки ONNX).

ONNX 1.15+.

Структура проекта
text
.
├── tensorrt_docker.dockerfile   # TensorRT 10.x
├── README.md
│
├── onnx/                        # монтируется → /workspace/onnx (входные ONNX)
├── runs/                        # монтируется → /workspace/runs (engine файлы)
└── logs/                        # монтируется → /workspace/logs (логи)
Сборка и запуск
bash
docker build -f tensorrt_docker.dockerfile -t tensorrt-opt .

# Сборка engine (FP16)
docker run -it --rm --gpus all --shm-size=8g \
    -v $(pwd)/onnx:/workspace/onnx \
    -v $(pwd)/runs:/workspace/runs \
    -v $(pwd)/logs:/workspace/logs \
    tensorrt-opt python /workspace/build_engine.py \
        --onnx /workspace/onnx/model.onnx \
        --engine /workspace/runs/model.engine \
        --precision fp16

# Бенчмарк
docker run -it --rm --gpus all --shm-size=8g \
    -v $(pwd)/onnx:/workspace/onnx \
    -v $(pwd)/runs:/workspace/runs \
    -v $(pwd)/logs:/workspace/logs \
    tensorrt-opt python /workspace/benchmark.py \
        --engine /workspace/runs/model.engine \
        --onnx /workspace/onnx/model.onnx
Режимы точности
Режим	Точность	Скорость	Когда использовать
FP32	Максимальная	Минимальная	Если критична точность
FP16	Хорошая	2-3x быстрее FP32	Рекомендуется по умолчанию
INT8	Сниженная	3-5x быстрее FP32	Максимальная скорость, нужен calibration dataset
Рекомендация: начинай с FP16 — это баланс точности и скорости. Если mAP/mIoU падает незначительно (≤0.5%), а latency критична — пробуй INT8.

INT8 требует calibration dataset — репрезентативная выборка изображений (100–500 штук), на которой TensorRT калибрует диапазоны активаций. Без калибровки INT8 даст плохие результаты.

Переменные окружения
build_engine.py
Переменная	Обязательна	Дефолт	Описание
--onnx	да	—	Путь к входному ONNX
--engine	нет	/workspace/runs/model.engine	Куда сохранить engine
--precision	нет	fp16	fp32 / fp16 / int8
--workspace	нет	4096	Workspace в MB (для оптимизации)
--log	нет	/workspace/logs/tensorrt.log	Лог сборки
benchmark.py
Переменная	Обязательна	Дефолт	Описание
--engine	да	—	Путь к engine
--onnx	да	—	Путь к ONNX (для сравнения)
--shape	нет	1,3,640,640	Input shape для бенчмарка
--log	нет	/workspace/logs/benchmark.log	Лог бенчмарка
Что на выходе
Модели
text
runs/
└── model.engine      # TensorRT engine (сериализованный)
Логи
text
logs/
├── tensorrt.log      # лог сборки engine (build time, precision)
└── benchmark.log     # сравнение latency: TensorRT vs ONNX Runtime
Пример benchmark.log
text
=== Benchmark ===
TensorRT: 12.34ms (p95=13.21ms)
ONNX RT:  28.56ms (p95=31.02ms)
Speedup: 2.31x
Как читать результат бенчмарка
Speedup > 2x — TensorRT работает хорошо, есть смысл деплоить.

Speedup 1.2–2x — умеренное ускорение, оценивай по задаче.

Speedup < 1.2x — модель маленькая или уже оптимизирована, ONNX Runtime может быть достаточно.

p95 vs mean — если p95 сильно выше mean, есть jitter (конкуренция за GPU, memory allocation).

Диагностика проблем
ONNX parse failed
Несовместимая версия ONNX или opset. Решение:

Попробуй экспортировать ONNX с opset=13 (TensorRT лучше работает с 13–17).

Проверь, что ONNX валиден: python -c "import onnx; onnx.checker.check_model('model.onnx')".

Engine built, but inference gives wrong results
Несовместимость версий TensorRT. Решение: убедись, что версия TensorRT на машине сборки engine совпадает с версией на машине инференса.

CUDA out of memory при сборке engine
Уменьши --workspace (например, до 2048 MB). Большой workspace нужен для сложных моделей, но не всегда обязателен.

INT8 calibration failed
Нет calibration dataset или он нерепрезентативен. Решение: подготовь 100–500 изображений, покрывающих все классы и условия съёмки.

ImportError: pycuda
Не установлен pycuda. В контейнере он уже есть. Если ошибка сохраняется — проверь, что базовый образ не подменён на другой.

Engine deserialization failed
Несовпадение версий TensorRT. Решение: пересобери engine на целевой машине.

Permission denied при записи в runs/ или logs/
Docker создал папки с правами root. Решение:

bash
sudo chown -R $USER:$USER runs logs
Создавай папки до первого docker run.

Контейнер не видит GPU
bash
docker run --gpus all nvidia/cuda:12.6-base nvidia-smi
Если не работает — проблема в nvidia-container-toolkit.

Какую модель выбирать
Задача	Рекомендация
Production deployment ONNX-моделей	TensorRT FP16
Максимальная скорость на NVIDIA GPU	TensorRT INT8 (с calibration)
Критична точность	TensorRT FP32 или оставить ONNX Runtime
Маленькие модели (<10M params)	ONNX Runtime может быть достаточно
Трансформеры (DETR, ViT)	TensorRT FP16 — хорошо оптимизирует attention
Совместимость с другими контейнерами
TensorRT работает с ONNX-моделями, экспортированными из любого из контейнеров этой коллекции:

Источник			ONNX-файл	Работает с TensorRT
YOLOv9				model.onnx	✅ Да
RF-DETR				model.onnx	✅ Да
D-FINE				model.onnx	✅ Да
MMDetection			model.onnx	✅ Да
MMDetection3D		model.onnx	⚠️ Ограниченно (sparse convolutions)
Detectron2			model.onnx	✅ Да
Double Head			model.onnx	✅ Да
DeepLabV3+ 			model.onnx	✅ Да
U-Net 				model.onnx	✅ Да
SegFormer 			model.onnx	✅ Да
PIDNet 				model.onnx	✅ Да
Mask2Former			model.onnx	✅ Да
timm				model.onnx	✅ Да
MMPretrain			model.onnx	✅ Да
RTMPose 			model.onnx	✅ Да
MMPose				model.onnx	✅ Да
Depth Anything V2   model.onnx	✅ Да
MiDaS				model.onnx	✅ Да
Real-ESRGAN 		model.onnx	✅ Да
MMagic				model.onnx	✅ Да
FLUX 				model.onnx	✅ Да
SDXL				model.onnx	⚠️ Требует кастомного экспорта
Рекомендация: для production-деплоя — комбинируй свой контейнер (например, YOLOv9) + TensorRT для финальной оптимизации.

Лицензии
Модель			Лицензия	Коммерческое использование
TensorRT 10.x	NVIDIA SLA	✅ Да (бесплатно, только для NVIDIA GPU)

TensorRT можно использовать в закрытых коммерческих продуктах без раскрытия кода. 
Runtime files распространяемы, SDK лицензирован только для систем с NVIDIA GPU.

