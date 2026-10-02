### CV Pipeline: полная коллекция Docker-контейнеров для Computer Vision
41 Docker-контейнер для 22 задач Computer Vision с permissive-лицензиями (MIT / Apache 2.0), позволяющими коммерческое использование без раскрытия кода. Каждый контейнер — это полностью автоматический пайплайн: один docker run → обучение, тестирование, экспорт в ONNX, повторный тест, графики и логи.

Содержание^
- Быстрый обзор
- Структура репозитория
- Что в каждой папке
- Общие требования
- Быстрый старт
- Общие принципы
- Лицензии

## Быстрый обзор
```
|---------------------------|-----------------------------------------------------|:-----------:|-------------------------------------------------------------------------|
|            Папка          |                         Задача                      | Контейнеров |                             Ключевые модели                             |
|---------------------------|-----------------------------------------------------|:-----------:|-------------------------------------------------------------------------|
| `my_detectors`            | Детекция объектов (2D и 3D)                         |     7       | YOLOv9, RF-DETR, D-FINE, MMDet, MMDet3D, Detectron2 Faster, Double Head |
|---------------------------|-----------------------------------------------------|:-----------:|-------------------------------------------------------------------------|
| `my_segmentators`         | Семантическая и instance-сегментация                |     7       | DeepLabV3+, U-Net, SegFormer,<br>PIDNet, Mask2Former, Mask R-CNN, MONAI |
|---------------------------|-----------------------------------------------------|:-----------:|-------------------------------------------------------------------------|
| `my_recognizers`          | Распознавание текста (OCR)                          |     3       | LightOnOCR, PaddleOCR, MMOCR                                            |
|---------------------------|-----------------------------------------------------|:-----------:|-------------------------------------------------------------------------|
| `my_classification_...`   | Классификация, anomaly detection, GAN               |     5       | timm, MMPretrain, Anomalib, PatchCore, BigGAN                           |
|---------------------------|-----------------------------------------------------|:-----------:|-------------------------------------------------------------------------|
| `my_tracking_pose_...`    | Трекинг, pose, keypoints                            |     6       | ByteTrack, MMTracking, MMPose, RTMPose, (MMPose, Detectron2)-Keypoint   |
|---------------------------|-----------------------------------------------------|:-----------:|-------------------------------------------------------------------------|
| `my_depth_estimation...`  | Depth, restoration, face, docs, видео, генерация    |     12      | Depth Anything V2, MiDaS, Real-ESRGAN, MMagic, UniFace, RetinaFace      |
|                           |                                                     |             | LayoutParser, Table Transformer, MMAction2, VideoMAE, FLUX.2[klein4B]   |
|                           |                                                     |             | SDXL                                                                    |
|---------------------------|-----------------------------------------------------|:-----------:|-------------------------------------------------------------------------|
| `my_optimization`         | Оптимизация инференса                               |     1       | TensorRT 10.x                                                           |
|---------------------------|-----------------------------------------------------|:-----------:|-------------------------------------------------------------------------|
```
Итого: 41 контейнер.

## Структура репозитория
text
.
├── my_detectors/
│   ├── yolov9_docker.dockerfile
│   ├── rfdetr_docker.dockerfile
│   ├── dfine_docker.dockerfile
│   ├── mmdet_yolox_docker.dockerfile
│   ├── mmdetection3d_docker.dockerfile
│   ├── detectron2_docker.dockerfile
│   ├── double_head_docker.dockerfile
│   └── README.md
│
├── my_segmentators/
│   ├── deeplabv3plus_docker.dockerfile
│   ├── unet_docker.dockerfile
│   ├── segformer_docker.dockerfile
│   ├── pidnet_docker.dockerfile
│   ├── mask2former_docker.dockerfile
│   ├── detectron2_maskrcnn_docker.dockerfile
│   ├── monai_docker.dockerfile
│   └── README.md
│
├── my_recognizers/
│   ├── lightonocr_docker.dockerfile
│   ├── paddleocr_docker.dockerfile
│   ├── mmocr_docker.dockerfile
│   └── README.md
│
├── my_classification_anomaly_detection/
│   ├── timm_docker.dockerfile
│   ├── mmpretrain_docker.dockerfile
│   ├── anomalib_docker.dockerfile
│   ├── patchcore_docker.dockerfile
│   ├── biggan_docker.dockerfile
│   └── README.md
│
├── my_tracking_pose_estimation_keypoint_detection/
│   ├── bytetrack_docker.dockerfile
│   ├── mmtracking_docker.dockerfile
│   ├── mmpose_docker.dockerfile
│   ├── rtmpose_docker.dockerfile
│   ├── mmpose_keypoint_docker.dockerfile
│   ├── detectron2_keypoint_docker.dockerfile
│   └── README.md
│
├── my_depth_estimation_image_restoration_face_recognition_KIELayout_video_understanding/
│   ├── depth_anything_docker.dockerfile
│   ├── midas_docker.dockerfile
│   ├── real_esrgan_docker.dockerfile
│   ├── mmagic_docker.dockerfile
│   ├── uniface_docker.dockerfile
│   ├── retinaface_docker.dockerfile
│   ├── layoutparser_docker.dockerfile
│   ├── table_transformer_docker.dockerfile
│   ├── mmaction2_docker.dockerfile
│   ├── videomae_docker.dockerfile
│   ├── flux_docker.dockerfile
│   ├── stable_diffusion_docker.dockerfile
│   └── README.md
│
├── my_optimization/
│   ├── tensorrt_docker.dockerfile
│   └── README.md
│
└── README.md                    # этот файл
Что в каждой папке
📦 my_detectors — детекция объектов
Зачем: найти объекты на изображении (bbox + класс).

Что внутри:

YOLOv9 (MIT) — классика one-stage, знакомая архитектура.

RF-DETR (Apache 2.0) — SOTA one-stage DETR без NMS.

D-FINE (Apache 2.0) — лёгкий one-stage DETR для edge.

MMDetection (Apache 2.0) — модульный конструктор: перебор backbone/neck/head через env.

MMDetection3D (Apache 2.0) — 3D-детекция на LiDAR/monocular/multi-modal.

Detectron2 Faster R-CNN (Apache 2.0) — two-stage с гибким PyTorch-кодом.

Double Head R-CNN (Apache 2.0) — two-stage с раздельными головами (+3.5 AP).

Когда выбирать:

Real-time → YOLOv9, MMDetection YOLOX-S.

Максимальное качество → RF-DETR.

Мелкие объекты → Double Head, Detectron2 Faster.

3D-детекция → MMDetection3D.

Эксперименты с архитектурой → MMDetection.

📦 my_segmentators — сегментация
Зачем: классифицировать каждый пиксель (semantic) или различать экземпляры (instance).

Что внутри:

DeepLabV3+ (Apache 2.0) — semantic, чёткиe границы (82.1 mIoU).

U-Net (Apache 2.0) — лёгкий, для малых данных и медицины.

SegFormer (Apache 2.0) — трансформерный semantic.

PIDNet (Apache 2.0) — real-time semantic (93 FPS).

Mask2Former (Apache 2.0) — универсальный (semantic + instance + panoptic).

Detectron2 Mask R-CNN (Apache 2.0) — instance-сегментация.

MONAI (Apache 2.0) — медицинская 3D-сегментация (DICOM/NIfTI).

Когда выбирать:

Real-time → PIDNet.

Точность → Mask2Former, SegFormer.

Медицина → U-Net, MONAI.

Instance → Mask R-CNN.

📦 my_recognizers — распознавание текста (OCR)
Зачем: извлечь текст из изображений документов, сканов, чеков.

Что внутри:

LightOnOCR-2-1B (Apache 2.0) — end-to-end VLM, SOTA на OlmOCR-Bench.

PaddleOCR (Apache 2.0) — традиционный toolkit (детектор + распознаватель), 80+ языков.

MMOCR (Apache 2.0) — модульный конструктор (CRNN/SAR/SEG), word-level метрики.

Когда выбирать:

End-to-end документы → LightOnOCR.

Промышленный OCR с ONNX → PaddleOCR.

Модульность, CRNN → MMOCR.

📦 my_classification_anomaly_detection — классификация, аномалии, GAN
Зачем: определить класс объекта, найти аномалии, генерировать изображения.

Что внутри:

timm (Apache 2.0) — 800+ моделей классификации.

MMPretrain (Apache 2.0) — классификация от OpenMMLab.

Anomalib (Apache 2.0) — anomaly detection, 15+ алгоритмов (PatchCore: 0.980 AUROC).

PatchCore (Apache 2.0) — standalone-реализация PatchCore.

BigGAN (MIT) — class-conditional GAN, FID 36.94.

Когда выбирать:

Классификация быстро → timm.

Классификация OpenMMLab → MMPretrain.

Аномалии production → Anomalib.

Аномалии кастомизация → PatchCore.

Class-conditional генерация → BigGAN.

📦 my_tracking_pose_estimation_keypoint_detection — трекинг, pose, keypoints
Зачем: отслеживать объекты во времени, находить ключевые точки тела/объектов.

Что внутри:

ByteTrack (MIT) — MOT поверх готовых детекций.

MMTracking (Apache 2.0) — модульный MOT/SOT/VID.

MMPose (Apache 2.0) — pose estimation (body/face/hand/animal).

RTMPose (Apache 2.0) — real-time multi-person pose.

MMPose Keypoint (Apache 2.0) — custom keypoints (multi-class).

Detectron2 Keypoint (Apache 2.0) — keypoint R-CNN (one-class).

Когда выбирать:

Трекинг production → ByteTrack.

Трекинг OpenMMLab → MMTracking.

Pose модульность → MMPose.

Pose real-time → RTMPose.

Custom keypoints → MMPose Keypoint / Detectron2 Keypoint.

📦 my_depth_estimation_image_restoration_face_recognition_KIELayout_video_understanding — depth, restoration, face, документы, видео, генерация
Зачем: оценка глубины, восстановление изображений, распознавание лиц, анализ документов, video understanding, генерация.

Что внутри:

Depth Anything V2 (Apache 2.0) — SOTA monocular depth.

MiDaS (MIT) — лёгкий depth estimation.

Real-ESRGAN (Apache 2.0) — blind super-resolution.

MMagic (Apache 2.0) — restoration toolkit (ESRGAN, BasicVSR).

UniFace (MIT) — all-in-one face analysis (детекция + распознавание + landmarks).

RetinaFace (MIT) — только детекция лиц.

LayoutParser (Apache 2.0) — документ layout analysis.

Table Transformer (MIT) — detection + structure recognition таблиц.

MMAction2 (Apache 2.0) — video understanding.

VideoMAE (Apache 2.0) — self-supervised видео.

FLUX.2 [klein] 4B (Apache 2.0) — text-to-image + editing, sub-second.

Stable Diffusion XL (CreativeML Open RAIL++-M) — максимальная кастомизация.

Когда выбирать:

Depth SOTA → Depth Anything V2.

Depth легко → MiDaS.

Restoration real-world → Real-ESRGAN.

Face полный пайплайн → UniFace.

Документы layout → LayoutParser.

Таблицы → Table Transformer.

Video → MMAction2, VideoMAE.

Генерация + скорость → FLUX.2 [klein] 4B.

Генерация + кастомизация → SDXL.

📦 my_optimization — оптимизация инференса
Зачем: ускорить ONNX-модели на NVIDIA GPU.

Что внутри:

TensorRT 10.x (NVIDIA SLA, бесплатно) — компилятор ONNX в engine. 2-5x ускорение.

Когда выбирать:

Production deployment ONNX-моделей.

Latency-critical задачи.

FP16/INT8 квантизация.

Общие требования
GPU и драйверы
Убедись, что nvidia-container-toolkit установлен и GPU доступен Docker'у:

bash
docker run --gpus all nvidia/cuda:12.6-base nvidia-smi
Если команда не выводит таблицу с GPU — внутри контейнера GPU не увидится, и пайплайн упадёт на этапе тренировки.

Shared memory
Все контейнеры требуют --shm-size=8g минимум. Для 3D-задач (MMDetection3D) и видео — 16g. Без этого DataLoader с num_workers > 0 упадёт с Bus error.

Диск
Запас свободного места на хосте — минимум 20 GB на датасет + чекпоинты. Для 3D-задач — 50 GB.

Быстрый старт
bash
# 1. Подготовка директорий
mkdir -p datasets runs logs work_dirs output

# 2. Выбор папки и модели
cd my_detectors

# 3. Сборка образа (пример для YOLOv9)
docker build -f yolov9_docker.dockerfile -t yolov9-mit .

# 4. Запуск пайплайна
docker run -it --rm --gpus all --shm-size=8g \
    -v $(pwd)/../datasets:/workspace/datasets \
    -v $(pwd)/../runs:/workspace/runs \
    -v $(pwd)/../logs:/workspace/logs \
    -e DATASET_CONFIG=/workspace/datasets/mydata.yaml \
    -e RUN_NAME=yolov9-mydata \
    yolov9-mit

# 5. Результат
ls ../logs/plots/                       # графики
ls ../runs/train/yolov9-mydata/weights/ # best.pt и best.onnx
Подробности по каждой модели — в README соответствующей папки.

Общие принципы
Единый пайплайн
Все контейнеры следуют одной схеме:

text
convert → train → test .pth → export .onnx → test .onnx → plots
convert — конвертация LabelMe/COCO/YOLO в формат модели.

train — обучение с параметрами из env.

test .pth — baseline mAP/mIoU на тестовой выборке.

export .onnx — экспорт для production.

test .onnx — проверка, что экспорт не сломал качество.

plots — графики: training curves, PR-curves (AUC-PR), confusion matrix, confidence profile.

Единая структура логов
text
logs/
├── 00_pipeline.log      # мастер-лог: все шаги с таймстампами и exit code
├── 01_train.log         # тренировка
├── 02_test_pth.log      # mAP/mIoU .pth
├── 03_export.log        # экспорт в ONNX
├── 04_test_onnx.log     # mAP/mIoU .onnx
└── plots/
    ├── 01_training_curves.png
    ├── 02_pr_curves.png
    ├── 03_confusion_matrix.png
    └── 04_confidence_profile.png
Первый файл при проблеме — 00_pipeline.log. Там видно, на каком шаге упало и с каким кодом.

Единый запуск
bash
docker run -it --rm --gpus all --shm-size=8g \
    -v $(pwd)/datasets:/workspace/datasets \
    -v $(pwd)/runs:/workspace/runs \
    -v $(pwd)/logs:/workspace/logs \
    -e <параметры> \
    <image>
Лицензии
Все 41 контейнер — commercial-clean. Можно использовать в закрытых коммерческих продуктах без раскрытия кода.

Тип лицензии	Модели
MIT	YOLOv9, ByteTrack, MiDaS, UniFace, RetinaFace, Table Transformer, BigGAN
Apache 2.0	RF-DETR, D-FINE, MMDetection, MMDetection3D, Detectron2 (все), Double Head, DeepLabV3+, U-Net, SegFormer, PIDNet, Mask2Former, MONAI, LightOnOCR, PaddleOCR, MMOCR, timm, MMPretrain, Anomalib, PatchCore, MMTracking, MMPose, RTMPose, Depth Anything V2, Real-ESRGAN, MMagic, LayoutParser, MMAction2, VideoMAE, FLUX.2 [klein] 4B
CreativeML Open RAIL++-M	Stable Diffusion XL (коммерчески разрешена с ограничениями)
NVIDIA SLA	TensorRT (бесплатно, только для NVIDIA GPU)
