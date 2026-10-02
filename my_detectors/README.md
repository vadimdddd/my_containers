CV Detection Pipeline: 7 моделей для детекции объектов
Семь Docker-контейнеров для задач детекции объектов с permissive-лицензиями, позволяющими коммерческое использование без раскрытия кода.

Каждый контейнер — это полностью автоматический пайплайн:

text
convert → train → test .pth → export .onnx → test .onnx → plots
Один docker run → на выходе готовая .onnx модель, логи и 4 графика.

Содержание
Обзор моделей

Структура проекта

Требования

Быстрый старт

Подготовка данных

Подготовка директорий

YOLOv9 (MIT)

RF-DETR (Apache 2.0)

D-FINE (Apache 2.0)

MMDetection: модульный конструктор (Apache 2.0)

MMDetection3D: 3D-детекция (Apache 2.0)

Detectron2: Faster R-CNN (Apache 2.0)

Double Head R-CNN (Apache 2.0)

Какую модель выбирать

Описание блоков кода

Что на выходе

Диагностика проблем

Лицензии

Обзор моделей
#	Модель	Тип	Лицензия	Формат данных	Особенность
1	YOLOv9	One-stage CNN	MIT	YOLO	Классика, знакомая архитектура
2	RF-DETR	One-stage DETR	Apache 2.0	COCO	SOTA качество, без NMS
3	D-FINE	One-stage DETR	Apache 2.0	COCO	Лёгкая, предсказуемая latency
4	MMDetection	Модульный конструктор	Apache 2.0	LabelMe→COCO	Перебор backbone/neck/head
5	MMDetection3D	3D-детекция	Apache 2.0	KITTI	LiDAR + monocular + multi-modal
6	Detectron2	Two-stage (Faster R-CNN)	Apache 2.0	COCO	Гибкий PyTorch-код
7	Double Head R-CNN	Two-stage	Apache 2.0	LabelMe→COCO	+3.5 AP над Faster R-CNN
Структура проекта
text
.
├── yolov9_docker.dockerfile          # YOLOv9 (MIT)
├── rfdetr_docker.dockerfile          # RF-DETR (Apache 2.0)
├── dfine_docker.dockerfile           # D-FINE (Apache 2.0)
├── mmdet_yolox_docker.dockerfile     # MMDetection модульный (Apache 2.0)
├── mmdetection3d_docker.dockerfile   # MMDetection3D (Apache 2.0)
├── detectron2_docker.dockerfile      # Detectron2 Faster R-CNN (Apache 2.0)
├── double_head_docker.dockerfile     # Double Head R-CNN (Apache 2.0)
├── README.md
│
├── datasets/                         # монтируется → /workspace/datasets
├── runs/                             # монтируется → /workspace/runs (ONNX, артефакты)
├── work_dirs/                        # монтируется → /workspace/work_dirs (MMDetection, MMDetection3D)
├── output/                           # монтируется → /workspace/output (Detectron2)
└── logs/                             # монтируется → /workspace/logs (логи, графики)
    └── plots/                        # графики PNG
Требования
GPU и драйверы
Убедись, что nvidia-container-toolkit установлен и GPU доступен Docker'у:

bash
docker run --gpus all nvidia/cuda:12.6-base nvidia-smi
Если команда не выводит таблицу с GPU — внутри контейнера GPU не увидится, и пайплайн упадёт на этапе тренировки. Установи nvidia-container-toolkit по официальной инструкции и перезапусти Docker daemon.

Shared memory
Все контейнеры требуют --shm-size=8g минимум. Для MMDetection3D рекомендуется --shm-size=16g (работа с 3D-данными). Без этого DataLoader с num_workers > 0 упадёт с Bus error при больших батчах.

Диск
Запас свободного места на хосте — минимум 20 GB на датасет + чекпоинты. Для MMDetection3D — минимум 50 GB (KITTI датасет объёмный). Two-stage модели (Detectron2, Double Head) сохраняют больше чекпоинтов, чем one-stage.

Версии (уже зафиксированы в образах)
Модель	PyTorch	CUDA	Базовый образ
YOLOv9 (MIT)	1.10.0	11.3	pytorch/pytorch:1.10.0-cuda11.3-cudnn8-devel
RF-DETR	2.6.0	12.6	pytorch/pytorch:2.6.0-cuda12.6-cudnn9-devel
D-FINE	2.6.0	12.6	pytorch/pytorch:2.6.0-cuda12.6-cudnn9-devel
MMDetection	1.10.0	11.3	pytorch/pytorch:1.10.0-cuda11.3-cudnn8-devel
MMDetection3D	2.6.0	12.6	pytorch/pytorch:2.6.0-cuda12.6-cudnn9-devel
Detectron2	1.10.0	11.3	pytorch/pytorch:1.10.0-cuda11.3-cudnn8-devel
Double Head	1.10.0	11.3	pytorch/pytorch:1.10.0-cuda11.3-cudnn8-devel
⚠️ YOLOv9, MMDetection, Detectron2, Double Head жёстко завязаны на свои версии PyTorch. YOLOv9 — на torchvision.ops.nms из 0.11.0 и GradScaler из 1.10.0. MMDetection 2.27.0 — на mmcv-full 1.7.0 под torch 1.10.0 + cu113. Detectron2 v0.6 — на pre-built wheel под torch 1.10.0 + cu113. Не меняй базовые образы.

⚠️ MMDetection3D v1.0.0rc5 требует MMCV 2.x + MMDetection 3.x + PyTorch 2.x + spconv 2.x. Это другая ветка OpenMMLab, несовместимая с MMDetection 2.27.0. Не путай образы.

Быстрый старт
bash
# 1. Подготовка директорий
mkdir -p datasets runs logs work_dirs output

# 2. Сборка образа (пример для YOLOv9)
docker build -f yolov9_docker.dockerfile -t yolov9-mit .

# 3. Запуск пайплайна
docker run -it --rm --gpus all --shm-size=8g \
    -v $(pwd)/datasets:/workspace/datasets \
    -v $(pwd)/runs:/workspace/runs \
    -v $(pwd)/logs:/workspace/logs \
    -e DATASET_CONFIG=/workspace/datasets/mydata.yaml \
    -e MODEL_CFG=models/detect/yolov9-c.yaml \
    -e PRETRAINED_WEIGHTS=weights/yolov9-c.pt \
    -e BATCH=4 -e EPOCHS=300 -e DEVICE=0 \
    -e RUN_NAME=yolov9-mydata \
    -e TEST_ANNOTATIONS=/workspace/datasets/test/_annotations.coco.json \
    -e TEST_IMAGES=/workspace/datasets/test \
    yolov9-mit

# 4. Результат
ls logs/plots/                          # графики
ls runs/train/yolov9-mydata/weights/    # best.pt и best.onnx
Подготовка данных
Формат датасета зависит от модели. Конвертируй данные заранее.

YOLOv9 — YOLO-формат
text
datasets/mydata/
├── mydata.yaml              # конфиг: пути к train/val/test + имена классов
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
Пример mydata.yaml:

yaml
path: /workspace/datasets/mydata
train: images/train
val: images/val
test: images/test

names:
  0: start
  1: defect
  2: joint
Каждому изображению соответствует .txt-файл с разметкой в формате class_id x_center y_center width height (нормализованные координаты).

RF-DETR и D-FINE — COCO-формат
text
datasets/mydata/
├── train/
│   ├── _annotations.coco.json
│   └── *.jpg
├── valid/
│   ├── _annotations.coco.json
│   └── *.jpg
└── test/
    ├── _annotations.coco.json
    └── *.jpg
Стандартный COCO-формат с полями images, annotations, categories. Roboflow экспортирует в этом формате «из коробки».

⚠️ D-FINE: в конфиге обязательно remap_mscoco_category: False, иначе категории будут маппиться на COCO-классы и обучение на кастомном датасете сломается.

MMDetection и Double Head — LabelMe JSON → VOC → COCO (автоматически)
Контейнеры сами конвертируют данные на стадии 0/6. От тебя нужно только:

text
datasets/
├── obj.names                    # имена классов, по одному на строку
├── dp6/
│   └── 10_115_137_131/
│       ├── 10_115_137_131_trainval/
│       │   ├── *.json           # LabelMe JSON
│       │   └── *.jpg
│       └── 10_115_137_131_test/
│           ├── *.json
│           └── *.jpg
└── dp7/
    └── ...
Скрипт конвертации:

Рекурсивно находит папки с *.json в datasets/.

Папки с test в имени → тестовая выборка, с trainval/train → обучающая.

Копирует JPG + генерирует VOC XML в mmdetection/data/nlmk_dp_data/.

Делает train/val split (по SPLIT_SIZE, дефолт 0.05).

MMDetection сам конвертирует VOC → COCO через pascal_voc.py.

Пример obj.names:

text
start
defect
joint
MMDetection3D — KITTI-формат
MMDetection3D работает с 3D-данными: LiDAR-облака, калибровки, 2D-изображения.

text
datasets/kitti/
├── ImageSets/
├── training/
│   ├── calib/*.txt              # калибровки
│   ├── image_2/*.png            # 2D-изображения
│   ├── label_2/*.txt            # 2D-разметка
│   └── velodyne/*.bin           # LiDAR-облака
└── testing/
    ├── calib/
    ├── image_2/
    └── velodyne/
Контейнер сам вызывает tools/create_data.py kitti для генерации info-файлов на стадии 0/3.

Detectron2 — COCO-формат (автоматически)
Detectron2-контейнер сам конвертирует LabelMe → COCO на стадии 0/6. Структура входа — та же, что у MMDetection. На выходе:

text
datasets/coco/
├── train/_annotations.coco.json + *.jpg
├── valid/_annotations.coco.json + *.jpg
└── test/_annotations.coco.json + *.jpg
Конвертация из других форматов
YOLO → COCO: через supervision (sv.DetectionDataset.from_yolo(...).as_coco()).

COCO → YOLO: через supervision (sv.DetectionDataset.from_coco(...).as_yolo()).

VOC → любой: supervision поддерживает from_pascal_voc.

LabelMe → VOC/COCO: встроено в MMDetection, Double Head и Detectron2 контейнеры.

Подготовка директорий
Создай папки на хосте заранее:

bash
mkdir -p datasets runs logs work_dirs output
Если этого не сделать, Docker создаст их при монтировании с правами root. В результате:

Писать в них из-под обычного пользователя будет нельзя.

Логи и чекпоинты будут принадлежать root.

Удалять их придётся через sudo rm -rf.

Все папки монтируются в контейнер:

Хост	Контейнер	Что внутри
./datasets	/workspace/datasets	Датасет
./runs	/workspace/runs	ONNX, промежуточные артефакты
./logs	/workspace/logs	Логи + графики в logs/plots/
./work_dirs	/workspace/work_dirs	Чекпоинты MMDetection, MMDetection3D, Double Head
./output	/workspace/output	Чекпоинты Detectron2
YOLOv9 (MIT)
Репозиторий: MultimediaTechLab/YOLO (MIT-сборка)
Веса: WongKinYiu/yolov9 releases (MIT)
Базовый образ: pytorch/pytorch:1.10.0-cuda11.3-cudnn8-devel

Сборка и запуск
bash
docker build -f yolov9_docker.dockerfile -t yolov9-mit .

docker run -it --rm --gpus all --shm-size=8g \
    -v $(pwd)/datasets:/workspace/datasets \
    -v $(pwd)/runs:/workspace/runs \
    -v $(pwd)/logs:/workspace/logs \
    -e DATASET_CONFIG=/workspace/datasets/mydata.yaml \
    -e MODEL_CFG=models/detect/yolov9-c.yaml \
    -e PRETRAINED_WEIGHTS=weights/yolov9-c.pt \
    -e BATCH=4 -e EPOCHS=300 -e DEVICE=0 \
    -e RUN_NAME=yolov9-mydata \
    -e TEST_ANNOTATIONS=/workspace/datasets/test/_annotations.coco.json \
    -e TEST_IMAGES=/workspace/datasets/test \
    yolov9-mit
Переменные окружения
Переменная	Обязательна	Дефолт	Описание
DATASET_CONFIG	да	—	Путь к data.yaml
MODEL_CFG	нет	models/detect/yolov9-c.yaml	YAML архитектуры
PRETRAINED_WEIGHTS	нет	weights/yolov9-c.pt	Стартовые веса
BATCH	нет	4	Batch size
EPOCHS	нет	300	Число эпох
DEVICE	нет	0	GPU id
RUN_NAME	нет	yolov9-custom	Имя запуска
IMG_SIZE	нет	640	Размер входного изображения
TEST_ANNOTATIONS	нет	datasets/test/_annotations.coco.json	COCO-аннотации для теста
TEST_IMAGES	нет	datasets/test	Папка тестовых изображений
RF-DETR (Apache 2.0)
Репозиторий: roboflow/rf-detr
Базовый образ: pytorch/pytorch:2.6.0-cuda12.6-cudnn9-devel

Сборка и запуск
bash
docker build -f rfdetr_docker.dockerfile -t rfdetr .

docker run -it --rm --gpus all --shm-size=8g \
    -v $(pwd)/datasets:/workspace/datasets \
    -v $(pwd)/runs:/workspace/runs \
    -v $(pwd)/logs:/workspace/logs \
    -e DATASET_DIR=/workspace/datasets/mydata \
    -e MODEL_SIZE=medium \
    -e EPOCHS=50 -e BATCH=4 \
    -e RUN_NAME=rfdetr-mydata \
    rfdetr
Переменные окружения
Переменная	Обязательна	Дефолт	Описание
DATASET_DIR	да	—	Корень COCO-датасета
MODEL_SIZE	нет	medium	base / medium / large
EPOCHS	нет	50	Число эпох
BATCH	нет	4	Batch size
RUN_NAME	нет	rfdetr	Имя запуска
D-FINE (Apache 2.0)
Репозиторий: Peterande/D-FINE
Базовый образ: pytorch/pytorch:2.6.0-cuda12.6-cudnn9-devel

Сборка и запуск
bash
docker build -f dfine_docker.dockerfile -t dfine .

docker run -it --rm --gpus all --shm-size=8g \
    -v $(pwd)/datasets:/workspace/datasets \
    -v $(pwd)/runs:/workspace/runs \
    -v $(pwd)/logs:/workspace/logs \
    -e CONFIG=configs/dfine/custom/dfine_hgnetv2_m_custom.yml \
    -e GPUS=1 \
    -e RUN_NAME=dfine-mydata \
    dfine
Переменные окружения
Переменная	Обязательна	Дефолт	Описание
CONFIG	да	—	Путь к YAML-конфигу датасета
GPUS	нет	1	Число GPU (для torchrun)
RUN_NAME	нет	dfine	Имя запуска
⚠️ Важно: в конфиге кастомного датасета обязательно remap_mscoco_category: False и правильное num_classes.

MMDetection: модульный конструктор (Apache 2.0)
Репозиторий: open-mmlab/mmdetection v2.27.0
Базовый образ: pytorch/pytorch:1.10.0-cuda11.3-cudnn8-devel

Что это даёт
MMDetection — платформа для сборки детекторов из независимых компонентов:

Компонент	Назначение	Варианты
Backbone	Извлечение признаков	CSPDarknet, ResNet, SwinTransformer, HRNet
Neck	Обработка между backbone и head	YOLOXPAFPN, FPN, PAFPN, NASFPN
Head	Предсказание bbox и классов	YOLOXHead, FCOSHead, ATSSHead, RetinaHead
Сборка и запуск
bash
docker build -f mmdet_yolox_docker.dockerfile -t mmdet-modular .

# Дефолт (YOLOX-S: CSPDarknet + YOLOXPAFPN + YOLOXHead)
docker run -it --rm --gpus all --shm-size=8g \
    -v $(pwd)/datasets:/workspace/datasets \
    -v $(pwd)/work_dirs:/workspace/work_dirs \
    -v $(pwd)/runs:/workspace/runs \
    -v $(pwd)/logs:/workspace/logs \
    -e RUN_NAME=yolox-default \
    mmdet-modular

# Своя комбинация (Swin + FPN + ATSS)
docker run -it --rm --gpus all --shm-size=8g \
    -v $(pwd)/datasets:/workspace/datasets \
    -v $(pwd)/work_dirs:/workspace/work_dirs \
    -v $(pwd)/runs:/workspace/runs \
    -v $(pwd)/logs:/workspace/logs \
    -e BACKBONE=SwinTransformer \
    -e NECK=FPN \
    -e HEAD=ATSSHead \
    -e NUM_CLASSES=3 \
    -e CLASSES="start defect joint" \
    -e RUN_NAME=swin-atss \
    mmdet-modular
Переменные окружения
Переменная	Обязательна	Дефолт	Описание
CONFIG	нет	—	Готовый конфиг. Если задан — генерация пропускается
BACKBONE	нет	CSPDarknet	CSPDarknet / ResNet / SwinTransformer / HRNet
NECK	нет	YOLOXPAFPN	YOLOXPAFPN / FPN / PAFPN / NASFPN
HEAD	нет	YOLOXHead	YOLOXHead / FCOSHead / ATSSHead / RetinaHead
NUM_CLASSES	нет	3	Число классов
CLASSES	нет	class1 class2 class3	Список классов через пробел
EPOCHS	нет	300	Число эпох
BATCH	нет	8	Batch size
WORKERS	нет	4	DataLoader workers
IMG_SIZE	нет	640	Размер входного изображения
GPUS	нет	1	Число GPU
RUN_NAME	нет	mmdet_run	Имя запуска
DATASETS_DIR	нет	/workspace/datasets	Корень с LabelMe JSON
DATA_MAIN_FOLDER	нет	nlmk_dp_data	Имя подпапки для VOC
OBJ_NAMES_FILE	нет	obj.names	Файл с классами
SPLIT_SIZE	нет	0.05	Доля val при split
RANDOM_STATE	нет	42	Seed для split
DATA_ROOT	нет	/workspace/mmdetection/data/VOC2007	COCO-датасет
Готовые комбинации (COCO mAP)
Комбинация	mAP	Скорость	Для чего
YOLOX-S (CSPDarknet + YOLOXPAFPN + YOLOXHead)	40.5%	⚡⚡⚡	Реалтайм, edge
ATSS-R50 (ResNet50 + FPN + ATSSHead)	39.3%	⚡⚡	Anchor-free baseline
FCOS-R50 (ResNet50 + FPN + FCOSHead)	38.5%	⚡⚡	Anchor-free, простой
RetinaNet-R50 (ResNet50 + FPN + RetinaHead)	36.4%	⚡⚡	Классика
Swin-ATSS (SwinTransformer + FPN + ATSSHead)	46.5%	⚡	Трансформерный SOTA
⚠️ Ограничения генератора
YOLOXHead совместим с CSPDarknet + YOLOXPAFPN — родная связка. С другими backbone может не сойтись по in_channels.

FCOSHead, ATSSHead, RetinaHead требуют 5 уровней FPN. Если backbone даёт 3 уровня (CSPDarknet), нужно править strides и in_channels.

SwinTransformer и HRNet не работают с YOLOXHead «из коробки».

Для нестандартной комбинации — используй CONFIG и монтируй свой .py-файл.

MMDetection3D: 3D-детекция (Apache 2.0)
Репозиторий: open-mmlab/mmdetection3d v1.0.0rc5
Базовый образ: pytorch/pytorch:2.6.0-cuda12.6-cudnn9-devel

Что это даёт
MMDetection3D — OpenMMLab фреймворк для 3D-детекции объектов:

Тип	Модели	Применение
LiDAR	PointPillars, SECOND, CenterPoint	Автономное вождение
Monocular	FCOS3D, PGD	3D по 2D-изображению
Multi-modal	BEVFusion, UniAD	LiDAR + camera fusion
Ключевые цифры:

CenterPoint — LiDAR-only, 0.53 mAP, 200 FPS на Jetson.

UniAD v2 — end-to-end (detection + tracking + planning), 0.393 AMOTA.

BEVFusion — multi-modal fusion, SOTA на nuScenes.

⚠️ Требует spconv 2.x для sparse convolutions (обязателен для SECOND, CenterPoint).

Сборка и запуск
bash
docker build -f mmdetection3d_docker.dockerfile -t mmdet3d .

docker run -it --rm --gpus all --shm-size=16g \
    -v $(pwd)/datasets:/workspace/datasets \
    -v $(pwd)/work_dirs:/workspace/work_dirs \
    -v $(pwd)/logs:/workspace/logs \
    -e CONFIG=configs/pointpillars/pointpillars_hv_secfpn_8xb6-160e_kitti-3d-3class.py \
    -e GPUS=1 \
    -e RUN_NAME=mmdet3d-kitti \
    mmdet3d
Переменные окружения
Переменная	Обязательна	Дефолт	Описание
CONFIG	нет	configs/pointpillars/pointpillars_hv_secfpn_8xb6-160e_kitti-3d-3class.py	Конфиг MMDetection3D
GPUS	нет	1	Число GPU (для dist_train)
RUN_NAME	нет	mmdet3d	Имя запуска
DATASETS_DIR	нет	/workspace/datasets	Корень с KITTI
MMDET3D_DATA	нет	/workspace/mmdetection3d/data	Куда писать info-файлы
Формат данных
KITTI-формат — стандарт для 3D-детекции. Включает:

calib/ — калибровочные матрицы (LiDAR ↔ camera).

image_2/ — 2D-изображения.

label_2/ — 2D-разметка (классы, bbox).

velodyne/ — LiDAR-облака (.bin).

Контейнер сам вызывает tools/create_data.py kitti для генерации .pkl info-файлов.

Когда выбирать
Автономное вождение — детекция машин, пешеходов, велосипедистов в 3D.

Робототехника — 3D-навигация, манипуляция.

AR/VR — 3D-реконструкция сцены.

Промышленность — контроль объёмных объектов, складская логистика.

Detectron2: Faster R-CNN (Apache 2.0)
Репозиторий: facebookresearch/detectron2 v0.6
Базовый образ: pytorch/pytorch:1.10.0-cuda11.3-cudnn8-devel

Что это даёт
Detectron2 — PyTorch-нативный two-stage детектор. Проще модифицировать, чем MMDetection: head/backbone/loss переписываются через наследование классов.

Сборка и запуск
bash
docker build -f detectron2_docker.dockerfile -t detectron2-faster .

docker run -it --rm --gpus all --shm-size=8g \
    -v $(pwd)/datasets:/workspace/datasets \
    -v $(pwd)/output:/workspace/output \
    -v $(pwd)/runs:/workspace/runs \
    -v $(pwd)/logs:/workspace/logs \
    -e NUM_CLASSES=3 \
    -e EPOCHS=12 \
    -e BATCH=2 \
    detectron2-faster
Переменные окружения
Переменная	Обязательна	Дефолт	Описание
NUM_CLASSES	нет	3	Число классов
EPOCHS	нет	12	Число эпох (аппроксимируется в MAX_ITER)
BATCH	нет	2	Batch size
WORKERS	нет	2	DataLoader workers
BASE_LR	нет	0.00025	Learning rate
MAX_ITER	нет	30000	Число итераций
MODEL_CFG	нет	COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml	Конфиг из Model Zoo
IMG_SIZE	нет	640	Размер входного изображения
COCO_DIR	нет	/workspace/datasets/coco	COCO-датасет
OUTPUT_DIR	нет	/workspace/output	Куда писать чекпоинты
Когда выбирать
Нужен один конкретный two-stage без экспериментов с архитектурой.

Хочешь модифицировать head/backbone через наследование классов PyTorch.

Готовые веса COCO для Faster R-CNN, Cascade R-CNN, Mask R-CNN.

Double Head R-CNN (Apache 2.0)
Репозиторий: open-mmlab/mmdetection v2.27.0
Базовый образ: pytorch/pytorch:1.10.0-cuda11.3-cudnn8-devel

Что такое Double Head
Идея из статьи «Rethinking Classification and Localization for Object Detection»:

fc-head (2 FC слоя) — лучше для классификации: пространственно-чувствителен, различает «полный объект vs часть».

conv-head (4 свёртки) — лучше для регрессии bbox: устойчив к пространственным вариациям.

Double Head совмещает: fc-head → classification, conv-head → bbox regression. Даёт +3.5 AP на ResNet-50 относительно Faster R-CNN.

Сборка и запуск
bash
docker build -f double_head_docker.dockerfile -t double-head .

docker run -it --rm --gpus all --shm-size=8g \
    -v $(pwd)/datasets:/workspace/datasets \
    -v $(pwd)/work_dirs:/workspace/work_dirs \
    -v $(pwd)/runs:/workspace/runs \
    -v $(pwd)/logs:/workspace/logs \
    -e NUM_CLASSES=3 \
    -e CLASSES="start defect joint" \
    -e EPOCHS=12 \
    -e BATCH=2 \
    -e RUN_NAME=double-head \
    double-head
Переменные окружения
Переменная	Обязательна	Дефолт	Описание
CONFIG	нет	—	Готовый конфиг (иначе генерируется)
NUM_CLASSES	нет	3	Число классов
CLASSES	нет	class1 class2 class3	Список классов
EPOCHS	нет	12	Число эпох
BATCH	нет	2	Batch size
WORKERS	нет	2	DataLoader workers
IMG_SIZE	нет	640	Размер входного изображения
RUN_NAME	нет	double_head	Имя запуска
Когда выбирать
Задача с мелкими объектами или объектами, у которых важна точная локализация границ.

Промышленный CV: дефекты, сварные швы, мелкие детали.

Готовность к меньшей скорости (two-stage медленнее one-stage в 3–5 раз).

Какую модель выбирать
По задаче
Задача	Рекомендация
Real-time детекция (≥30 FPS)	YOLOv9, MMDetection YOLOX-S
Максимальное качество на COCO	RF-DETR Medium
Лёгкая модель для edge	D-FINE Nano
Мелкие объекты в сложной сцене	Double Head, Detectron2 Faster R-CNN
Эксперименты с архитектурой	MMDetection (перебор backbone/neck/head)
Гибкая модификация head/loss	Detectron2
3D-детекция (LiDAR)	MMDetection3D (CenterPoint, PointPillars)
3D-детекция (monocular)	MMDetection3D (FCOS3D)
3D multi-modal fusion	MMDetection3D (BEVFusion, UniAD)
Сегментация + детекция	RF-DETR (RF-DETR-Seg)
LabelMe-разметка «из коробки»	MMDetection, Double Head, Detectron2
По типу модели
One-stage (YOLOv9, MMDetection one-stage, RF-DETR, D-FINE):

Быстрые, real-time.

Хороши на средних и крупных объектах.

Проще в тренировке, меньше VRAM.

Two-stage (Detectron2 Faster R-CNN, Double Head, MMDetection two-stage):

Точнее на мелких объектах и сложных границах.

Медленнее в 3–5 раз.

Больше VRAM из-за RoIAlign и второго этапа.

3D-детекция (MMDetection3D):

Работает с LiDAR-облаками и/или multi-view.

Требует KITTI/nuScenes-формат.

spconv 2.x для sparse convolutions.

По лицензии
Все семь моделей — permissive (MIT или Apache 2.0). Коммерческое использование без раскрытия кода разрешено.

Описание блоков кода
Все контейнеры следуют единой структуре. Ниже — что делает каждый скрипт.

00_convert.py — конвертация датасета
Что делает: читает LabelMe JSON из datasets/, генерирует Pascal VOC XML (для MMDetection) или COCO JSON (для Detectron2).

Ключевые функции:

find_subfolders_with_json(root) — рекурсивно ищет папки с .json, разделяет на trainval и test по имени.

correct_box_coords(box, h, w) — клипает bbox по границам изображения, меняет x1/x2 и y1/y2 если перепутаны.

main() — обходит все JSON, копирует JPG, пишет XML/COCO, делает train/val split.

Переменные env: DATASETS_DIR, OBJ_NAMES_FILE, SPLIT_SIZE, RANDOM_STATE.

00_prepare_data.py — подготовка KITTI (только MMDetection3D)
Что делает: вызывает tools/create_data.py kitti для генерации info-файлов из KITTI-датасета. Создаёт симлинк между datasets/kitti и mmdetection3d/data/kitti.

00b_voc_to_coco.sh — VOC → COCO (только MMDetection, Double Head)
Что делает: вызывает штатный tools/dataset_converters/pascal_voc.py из MMDetection. На выходе — mmdetection/data/VOC2007/ с COCO-аннотациями.

generate_config.py — генератор конфигов (MMDetection, Double Head)
Что делает: собирает .py-конфиг из выбранных BACKBONE + NECK + HEAD.

Ключевые функции:

BACKBONE_TEMPLATES, NECK_TEMPLATES, HEAD_TEMPLATES — словари с шаблонами компонентов.

render_dict(d, indent) — рекурсивно рендерит dict в python-литерал.

generate(args) — согласовывает in_channels между backbone, neck и head, пишет итоговый конфиг.

Логика: если CONFIG задан — используется готовый. Если нет — генерируется.

01_train.py / 01_train.sh — тренировка
YOLOv9: вызывает train_dual.py с параметрами из env.

RF-DETR / D-FINE: вызывает model.train() или train.py через torchrun.

MMDetection / Double Head: вызывает tools/train.py или tools/dist_train.sh.

MMDetection3D: вызывает tools/train.py или tools/dist_train.sh.

Detectron2: использует DefaultTrainer из detectron2.engine.

Все пишут лог в logs/01_train.log через tee -a.

02_test_pth.py / 02_test_pth.sh — тест .pth
Что делает: загружает обученный .pth, прогоняет тестовый набор, считает mAP через COCOeval.

Почему это baseline: с этим результатом сравнивается ONNX-версия. Если mAP .pth и .onnx отличаются >0.5% — проблема в экспорте.

03_export_onnx.py / 03_export_onnx.sh — экспорт в ONNX
YOLOv9: python export.py --weights best.pt --include onnx --simplify.

RF-DETR: model.export(output_dir=..., format="onnx") — нативно.

D-FINE: tools/deployment/export_onnx.py с патчем batch size.

MMDetection / Double Head: tools/deployment/pytorch2onnx.py, fallback на mmdeploy.

Detectron2: torch.onnx.export() с wrapper'ом для tensor-выхода.

MMDetection3D: экспорт в ONNX или TensorRT через mmdeploy.

04_test_onnx.py — тест .onnx
Что делает: загружает ONNX через onnxruntime, прогоняет тестовый набор, считает mAP через COCOeval.

Ключевые функции:

preprocess(img, imgsz) — ресайз + нормализация (у каждой модели своя: mean/std).

postprocess(outputs) — NMS или парсинг DETR-выхода.

main() — обход тестового набора, сбор results, COCOeval.

На выходе: logs/04_test_onnx.log с mAP@0.5, mAP@0.5:0.95, avg latency.

05_plots.py — графики
Что генерирует:

График	Что показывает	Источник данных
01_training_curves.png	Loss, mAP, P, R по эпохам	results.csv (YOLO) / лог тренировки
02_pr_curves.png	PR-кривые, AUC-PR = AP	pycocotools COCOeval
03_confusion_matrix.png	Какие классы путаются	supervision
04_confidence_profile.png	P/R/F1 vs confidence threshold	Ручной расчёт
AUC-PR — это не отдельная метрика, а и есть AP. pycocotools считает AP как площадь под PR-кривой.

pipeline.sh — главный entrypoint
Что делает:

Пишет в мастер-лог 00_pipeline.log каждый шаг с таймстампом.

Последовательно вызывает все стадии.

При ошибке фиксирует exit code и останавливается (set -e + trap).

Мастер-лог — первый файл, который нужно смотреть при проблеме:

text
[2026-03-30 14:23:11] [PIPELINE] STEP 2/6: Training started → logs/01_train.log
[2026-03-30 16:45:33] [PIPELINE] STEP 2/6: Training finished in 8532s
[2026-03-30 16:47:12] [PIPELINE] STEP 4/6: Export finished in 45s
[2026-03-30 16:47:12] [PIPELINE]   ✓ ONNX created: /workspace/runs/.../best.onnx
При падении — увидишь FAILED at step X/6 (exit=N).

Что на выходе
Модели
text
runs/
├── train/<RUN_NAME>/weights/     # YOLOv9, RF-DETR, D-FINE
│   ├── best.pt
│   ├── last.pt
│   └── best.onnx
├── onnx/model.onnx               # MMDetection, MMDetection3D, Detectron2, Double Head
└── ...
Чекпоинты MMDetection, MMDetection3D и Double Head лежат в work_dirs/<RUN_NAME>/.
Чекпоинты Detectron2 лежат в output/.

Логи
text
logs/
├── 00_pipeline.log      # мастер-лог: все шаги с таймстампами и exit code
├── 00_prepare.log       # подготовка KITTI (MMDetection3D)
├── 00_convert.log       # конвертация датасета
├── 00b_voc_to_coco.log  # VOC → COCO (MMDetection, Double Head)
├── 01_train.log         # тренировка
├── 02_test_pth.log      # mAP .pth
├── 03_export.log        # экспорт в ONNX
├── 04_test_onnx.log     # mAP .onnx
└── plots/
    ├── 01_training_curves.png
    ├── 02_pr_curves.png
    ├── 03_confusion_matrix.png
    └── 04_confidence_profile.png
Как читать логи
Первый файл при проблеме — 00_pipeline.log. Там видно, на каком шаге упало.

Сравнение .pth vs .onnx:

bash
grep "mAP@0.5" logs/02_test_pth.log logs/04_test_onnx.log
Если разница >0.5% — проблема в экспорте.

Диагностика проблем
Bus error при тренировке
Мало shared memory. Увеличь --shm-size до 16g или 32g. Для MMDetection3D рекомендуется 16g минимум.

ImportError: libGL.so.1 или libgthread-2.0.so.0
Не хватает системных библиотек для OpenCV. Проверь, что базовый образ не подменён на -runtime вместо -devel.

CUDA out of memory
Уменьши BATCH. Ориентиры:

YOLOv9-C при 640×640 на 8GB VRAM — BATCH=2

RF-DETR Medium — BATCH=2

D-FINE Medium — BATCH=4

MMDetection YOLOX-S — BATCH=4

MMDetection Swin-ATSS — BATCH=2

MMDetection3D PointPillars — BATCH=4

MMDetection3D CenterPoint — BATCH=2

Detectron2 Faster R-CNN — BATCH=2

Double Head — BATCH=2

ONNX mAP сильно ниже, чем .pth
Три частые причины:

Opset version. Попробуй opset=13 вместо 16/17.

Execution provider. Проверь, что ONNX Runtime использует GPU:

python
print(session.get_providers())  # CUDAExecutionProvider должен быть первым
Preprocessing. Убедись, что в 04_test_onnx.py тот же ресайз, нормализация и порядок каналов, что и в .pth-тесте.

MMDetection: экспорт ONNX падает с bbox_coder
Известная проблема YOLOX. В контейнере включён fallback через mmdeploy.

MMDetection: KeyError: 'xxx' is not in the models registry
Указан BACKBONE/NECK/HEAD, которого нет в MMDetection v2.27.0. Проверь список допустимых значений.

MMDetection: несовместимая комбинация backbone/neck/head
Генератор не проверяет совместимость каналов. Используй CONFIG с готовым конфигом.

MMDetection3D: ImportError: spconv
Не установлен spconv 2.x. В контейнере он ставится через pip install spconv-cu126. Если ошибка сохраняется — проверь совместимость версии spconv с CUDA.

MMDetection3D: FileNotFoundError: kitti_infos_train.pkl
Не запустилась подготовка данных. Проверь logs/00_prepare.log — там должен быть вызов create_data.py kitti. Если датасет в другом формате — конвертируй в KITTI заранее.

MMDetection3D: KeyError: 'LIDAR_TOP'
Проблема с calib-файлами. KITTI должен содержать calib/*.txt с корректными матрицами. Если датасет в nuScenes — используй соответствующий конфиг.

Detectron2: ONNX tracing падает
Detectron2 Faster R-CNN возвращает list of dicts. В контейнере используется wrapper, который приводит выход к тензорам. Если ошибка сохраняется — проверь версию torch.onnx.export и opset.

Permission denied при записи в runs/, logs/, work_dirs/, output/
Docker создал папки с правами root. Решение:

bash
sudo chown -R $USER:$USER runs logs work_dirs output
Создавай папки до первого docker run.

Контейнер не видит GPU
bash
docker run --gpus all nvidia/cuda:12.6-base nvidia-smi
Если не работает — проблема в nvidia-container-toolkit.

Тренировка падает на импортах
Подменили базовый образ на другую версию PyTorch. Проверь FROM в Dockerfile:

Для YOLOv9, MMDetection, Detectron2, Double Head — только pytorch/pytorch:1.10.0-cuda11.3-cudnn8-devel.

Для MMDetection3D — только pytorch/pytorch:2.6.0-cuda12.6-cudnn9-devel.

Лицензии
Модель			Лицензия кода	Лицензия весов	Коммерческое использование
YOLOv9			MIT					MIT				✅ Да
RF-DETR			Apache 2.0		Apache 2.0			✅ Да
D-FINE			Apache 2.0		Apache 2.0			✅ Да
MMDetection		Apache 2.0		Apache 2.0			✅ Да
MMDetection3D	Apache 2.0		Apache 2.0			✅ Да
Detectron2		Apache 2.0		Apache 2.0			✅ Да
Double Head		Apache 2.0		Apache 2.0			✅ Да
Все семь образов можно использовать в закрытых коммерческих продуктах без юридических рисков.