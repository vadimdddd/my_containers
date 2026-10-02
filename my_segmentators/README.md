# CV Segmentation Pipeline: 7 моделей для сегментации

Семь Docker-контейнеров для задач сегментации с **permissive-лицензиями**, позволяющими коммерческое использование без раскрытия кода.

Каждый контейнер — это **полностью автоматический пайплайн**:
convert → train → test .pth → export .onnx → test .onnx → plots

text

Один `docker run` → на выходе готовая `.onnx` модель, логи и графики.

---

## Содержание

- [Обзор моделей](#обзор-моделей)
- [Три типа сегментации](#три-типа-сегментации)
- [Структура проекта](#структура-проекта)
- [Требования](#требования)
- [Подготовка данных](#подготовка-данных)
- [DeepLabV3+](#deeplabv3)
- [U-Net](#u-net)
- [SegFormer](#segformer)
- [PIDNet](#pidnet)
- [Mask2Former](#mask2former)
- [Detectron2 Mask R-CNN](#detectron2-mask-r-cnn)
- [MONAI](#monai)
- [Какую модель выбирать](#какую-модель-выбирать)
- [Описание блоков кода](#описание-блоков-кода)
- [Что на выходе](#что-на-выходе)
- [Диагностика проблем](#диагностика-проблем)
- [Лицензии](#лицензии)

---

## Обзор моделей

| # | Модель | Тип | Лицензия | Скорость | mIoU/Dice | Для чего |
|---|---|---|---|---|---|---|
| 1 | **DeepLabV3+** | Semantic | Apache 2.0 | ⚡ | 82.1 mIoU | Чёткиe границы |
| 2 | **U-Net** | Semantic | Apache 2.0 | ⚡ | 69.1 mIoU | Малые данные, медицина |
| 3 | **SegFormer** | Semantic (transformer) | Apache 2.0 | ⚡ | 45.6 mIoU (ADE20K) | Сложные сцены |
| 4 | **PIDNet** | Semantic, real-time | Apache 2.0 | ⚡⚡⚡ (93 FPS) | 78.6–80.6 mIoU | Real-time, edge |
| 5 | **Mask2Former** | Semantic+Instance+Panoptic | Apache 2.0 | ⚡ | 80.4–83.5 mIoU | Универсальность, SOTA |
| 6 | **Detectron2 Mask R-CNN** | Instance | Apache 2.0 | ⚡ | mask AP | Различение экземпляров |
| 7 | **MONAI** | Medical 3D | Apache 2.0 | — | Dice | Медицина (DICOM/NIfTI) |

---

## Три типа сегментации

Прежде чем выбирать модель, важно понять, какую именно задачу ты решаешь.

### Semantic segmentation (DeepLabV3+, U-Net, SegFormer, PIDNet, Mask2Former)

**Что делает:** классифицирует **каждый пиксель** изображения в один из классов.

**Пример:** на фотографии улицы все пиксели размечены как «дорога», «машина», «человек», «здание». Два разных человека — оба класса «человек», модель их не различает.

**Метрика:** mIoU (mean Intersection over Union).

**Формат данных:** PNG-маски, где каждый пиксель = class_id. Background = 0, классы = 1..N.

**Когда использовать:**
- Промышленный контроль качества (дефекты, трещины, сварные швы).
- Медицинская визуализация (опухоли, органы).
- Автономное вождение (дорога, разметка, препятствия).
- Аэрофотосъёмка (здания, растительность, вода).

### Instance segmentation (Detectron2 Mask R-CNN)

**Что делает:** различает **отдельные экземпляры** объектов и создаёт маску для каждого.

**Пример:** на фотографии улицы модель находит «человек #1», «человек #2», «машина #1» — каждый со своей маской.

**Метрика:** mask AP (Average Precision).

**Формат данных:** COCO с полем `segmentation` (полигоны или RLE).

**Когда использовать:**
- Подсчёт объектов (сколько людей, машин, деталей).
- Робототехника (захват конкретного объекта).
- Медицина (сегментация отдельных клеток).

### Medical 3D segmentation (MONAI)

**Что делает:** сегментирует **3D-объёмы** медицинских данных (DICOM, NIfTI).

**Пример:** МРТ-снимок мозга, где каждый воксель размечен как «серое вещество», «белое вещество», «опухоль».

**Метрика:** Dice, Hausdorff Distance.

**Формат данных:** NIfTI (`.nii.gz`) или DICOM.

**Когда использовать:**
- Медицинская визуализация (МРТ, КТ, УЗИ).
- Радиология, онкология.
- **Не для промышленного CV на 2D-изображениях.**

---

## Структура проекта
.
├── deeplabv3plus_docker.dockerfile # DeepLabV3+ (Apache 2.0)
├── unet_docker.dockerfile # U-Net (Apache 2.0)
├── segformer_docker.dockerfile # SegFormer (Apache 2.0)
├── pidnet_docker.dockerfile # PIDNet (Apache 2.0)
├── mask2former_docker.dockerfile # Mask2Former (Apache 2.0)
├── detectron2_maskrcnn_docker.dockerfile # Mask R-CNN (Apache 2.0)
├── monai_docker.dockerfile # MONAI (Apache 2.0)
├── README.md
│
├── datasets/ # монтируется → /workspace/datasets
├── work_dirs/ # монтируется → /workspace/work_dirs (MMSeg)
├── output/ # монтируется → /workspace/output (Detectron2)
├── runs/ # монтируется → /workspace/runs (ONNX)
└── logs/ # монтируется → /workspace/logs
└── plots/ # графики PNG

text

---

## Требования

### GPU и драйверы

Убедись, что **`nvidia-container-toolkit` установлен** и GPU доступен Docker'у:

```bash
docker run --gpus all nvidia/cuda:12.6-base nvidia-smi
Если команда не выводит таблицу с GPU — внутри контейнера GPU не увидится, и пайплайн упадёт на этапе тренировки.

Shared memory
Все контейнеры требуют --shm-size=8g минимум. Без этого DataLoader с num_workers > 0 упадёт с Bus error при больших батчах. Если увидишь эту ошибку — увеличь до 16g или 32g.

Диск
Запас свободного места на хосте — минимум 30 GB на датасет + чекпоинты. Сегментаторы сохраняют больше промежуточных масок, чем детекторы.

Версии (уже зафиксированы в образах)
Модель	PyTorch	CUDA	Базовый образ
DeepLabV3+	1.10.0	11.3	pytorch/pytorch:1.10.0-cuda11.3-cudnn8-devel
U-Net	1.10.0	11.3	pytorch/pytorch:1.10.0-cuda11.3-cudnn8-devel
SegFormer	1.10.0	11.3	pytorch/pytorch:1.10.0-cuda11.3-cudnn8-devel
PIDNet	1.10.0	11.3	pytorch/pytorch:1.10.0-cuda11.3-cudnn8-devel
Mask2Former	1.10.0	11.3	pytorch/pytorch:1.10.0-cuda11.3-cudnn8-devel
Mask R-CNN	1.10.0	11.3	pytorch/pytorch:1.10.0-cuda11.3-cudnn8-devel
MONAI	24.08	12.x	nvcr.io/nvidia/pytorch:24.08-py3
⚠️ Первые шесть жёстко завязаны на MMSegmentation v0.30.0 + mmcv-full 1.7.0 + torch 1.10.0 + cu113. Не меняй базовый образ — иначе пайплайн сломается на импортах.

⚠️ Mask2Former требует mmdet. Без него падает с TypeError: Mask2FormerHead ... unexpected keyword argument 'in_channels'. В контейнере mmdet==2.27.0 уже установлен.

Подготовка данных
Semantic segmentation (DeepLabV3+, U-Net, SegFormer, PIDNet, Mask2Former)
Формат: PNG-маски, где каждый пиксель = class_id. Background = 0, классы = 1..N.

text
mmsegmentation/data/nlmk_seg/
├── img_dir/
│   ├── train/*.jpg
│   ├── val/*.jpg
│   └── test/*.jpg
└── ann_dir/
    ├── train/*.png    # маски
    ├── val/*.png
    └── test/*.png
Что нужно на входе (перед запуском контейнера):

text
datasets/
├── obj.names                    # имена классов, по одному на строку
├── dp6/
│   └── 10_115_137_131/
│       ├── 10_115_137_131_trainval/
│       │   ├── *.json           # LabelMe JSON с полигонами
│       │   └── *.jpg
│       └── 10_115_137_131_test/
│           ├── *.json
│           └── *.jpg
└── dp7/
    └── ...
Конвертация автоматическая. Dockerfile сам конвертирует LabelMe JSON → PNG-маски на стадии 0/6. Полигоны заливаются через cv2.fillPoly.

Пример obj.names:

text
start
defect
joint
Instance segmentation (Detectron2 Mask R-CNN)
Формат: COCO с полем segmentation (полигоны или RLE).

text
datasets/coco/
├── train/_annotations.coco.json + *.jpg
├── valid/_annotations.coco.json + *.jpg
└── test/_annotations.coco.json + *.jpg
Конвертация автоматическая. Detectron2-контейнер сам конвертирует LabelMe → COCO с полигонами.

Medical 3D segmentation (MONAI)
Формат: NIfTI (.nii.gz) или DICOM. 3D-данные или 2D-срезы.

text
datasets/medical/
├── images/*.nii.gz
└── labels/*.nii.gz
Конвертации нет — MONAI работает с медицинскими форматами напрямую. Нужны специализированные трансформы.

DeepLabV3+
Репозиторий: MMSegmentation v0.30.0
Архитектура: ResNet-101 (dilated) + ASPP + decoder.
Лицензия: Apache 2.0.

Что это даёт
DeepLabV3+ — это классика semantic segmentation. Encoder извлекает признаки через dilated convolutions, ASPP (Atrous Spatial Pyramid Pooling) захватывает контекст на разных масштабах, decoder восстанавливает разрешение и уточняет границы.

Результаты (Cityscapes): 82.1 mIoU на ResNet-101. Хорошо работает на объектах с чёткими границами — дорожная разметка, здания, промышленные дефекты.

Сборка и запуск
bash
docker build -f deeplabv3plus_docker.dockerfile -t deeplabv3plus .

docker run -it --rm --gpus all --shm-size=8g \
    -v $(pwd)/datasets:/workspace/datasets \
    -v $(pwd)/work_dirs:/workspace/work_dirs \
    -v $(pwd)/runs:/workspace/runs \
    -v $(pwd)/logs:/workspace/logs \
    -e NUM_CLASSES=3 \
    -e CONFIG=configs/deeplabv3plus/deeplabv3plus_r101-d8_512x512_40k_voc12aug.py \
    -e RUN_NAME=deeplabv3plus-mydata \
    deeplabv3plus
Переменные окружения
Переменная	Обязательна	Дефолт	Описание
NUM_CLASSES	нет	3	Число классов
CONFIG	нет	configs/deeplabv3plus/deeplabv3plus_r101-d8_512x512_40k_voc12aug.py	Конфиг MMSeg
DATASETS_DIR	нет	/workspace/datasets	Корень с LabelMe JSON
OBJ_NAMES_FILE	нет	obj.names	Файл с классами
SPLIT_SIZE	нет	0.05	Доля val при split
RANDOM_STATE	нет	42	Seed
RUN_NAME	нет	deeplabv3plus	Имя запуска
TRAINED_WEIGHTS	нет	—	Путь к .pth для теста
ONNX_PATH	нет	/workspace/runs/onnx/model.onnx	Куда писать ONNX
U-Net
Репозиторий: MMSegmentation v0.30.0
Архитектура: Encoder-decoder с skip connections.
Лицензия: Apache 2.0.

Что это даёт
U-Net — самая лёгкая из semantic-сегментаторов. Классическая U-образная форма: encoder сжимает изображение, decoder восстанавливает, skip connections передают детали с ранних слоёв.

Результаты: 69.1 mIoU на Cityscapes. 78.67 Dice на DRIVE (медицинские сосуды). Лёгкая — 17.91 GB памяти для обучения. Хороша для малых данных и медицинских задач с небольшим числом классов.

Сборка и запуск
bash
docker build -f unet_docker.dockerfile -t unet-seg .

docker run -it --rm --gpus all --shm-size=8g \
    -v $(pwd)/datasets:/workspace/datasets \
    -v $(pwd)/work_dirs:/workspace/work_dirs \
    -v $(pwd)/runs:/workspace/runs \
    -v $(pwd)/logs:/workspace/logs \
    -e NUM_CLASSES=3 \
    -e CONFIG=configs/unet/unet-s5-d16_fcn_4xb4-160k_cityscapes-512x1024.py \
    -e RUN_NAME=unet-mydata \
    unet-seg
Переменные окружения
Переменная	Обязательна	Дефолт	Описание
NUM_CLASSES	нет	3	Число классов
CONFIG	нет	configs/unet/unet-s5-d16_fcn_4xb4-160k_cityscapes-512x1024.py	Конфиг MMSeg
RUN_NAME	нет	unet	Имя запуска
Остальные переменные — как у DeepLabV3+.

SegFormer
Репозиторий: MMSegmentation v0.30.0
Архитектура: Mix Transformer (MiT) encoder + lightweight MLP decoder.
Лицензия: Apache 2.0.

Что это даёт
SegFormer — трансформерный сегментатор. Backbone MiT — иерархический трансформер, который извлекает признаки на нескольких масштабах. Decoder — простой MLP без сложных операций.

Результаты:

MIT-B0: 37.41 mIoU при 19.49 ms (V100) — самый лёгкий.

MIT-B2: 45.58 mIoU при 32.38 ms — баланс.

MIT-B5: 49.13 mIoU — точнее, но тяжелее.

Backbone: embed_dims=64, num_layers=[3, 4, 6, 3], decode_head in_channels=[64, 128, 320, 512]. Точнее U-Net на сложных сценах, но тяжелее.

Сборка и запуск
bash
docker build -f segformer_docker.dockerfile -t segformer .

docker run -it --rm --gpus all --shm-size=8g \
    -v $(pwd)/datasets:/workspace/datasets \
    -v $(pwd)/work_dirs:/workspace/work_dirs \
    -v $(pwd)/runs:/workspace/runs \
    -v $(pwd)/logs:/workspace/logs \
    -e NUM_CLASSES=3 \
    -e CONFIG=configs/segformer/segformer_mit-b2_8xb1-160k_cityscapes-1024x1024.py \
    -e RUN_NAME=segformer-mydata \
    segformer
Переменные окружения
Переменная	Обязательна	Дефолт	Описание
NUM_CLASSES	нет	3	Число классов
CONFIG	нет	configs/segformer/segformer_mit-b2_8xb1-160k_cityscapes-1024x1024.py	Конфиг MMSeg
RUN_NAME	нет	segformer	Имя запуска
Остальные переменные — как у DeepLabV3+.

PIDNet
Репозиторий: MMSegmentation v0.30.0 (официальная реализация в MMSeg)
Архитектура: три ветки P (detail), I (context), D (boundary). Вдохновлена PID-контроллером.
Лицензия: Apache 2.0.

Что это даёт
PIDNet — лучшее соотношение точность/скорость среди real-time сегментаторов.

Результаты (Cityscapes test):

PIDNet-S: 78.6 mIoU @ 93.2 FPS

PIDNet-M: 79.8 mIoU @ 42.2 FPS

PIDNet-L: 80.6 mIoU @ 31.1 FPS

Идея: предыдущие two-branch архитектуры (BiSeNet, DDRNet) страдают от overshoot — детальные предсказания «перебиваются» контекстными. PIDNet добавляет третью ветку boundary detection, которая гасит overshoot — как «D» (derivative) в PID-контроллере.

Конфиги в MMSeg:

configs/pidnet/pidnet-s_2xb6-120k_1024x1024-cityscapes.py

configs/pidnet/pidnet-m_2xb6-120k_1024x1024-cityscapes.py

configs/pidnet/pidnet-l_2xb6-120k_1024x1024-cityscapes.py

Сборка и запуск
bash
docker build -f pidnet_docker.dockerfile -t pidnet .

docker run -it --rm --gpus all --shm-size=8g \
    -v $(pwd)/datasets:/workspace/datasets \
    -v $(pwd)/work_dirs:/workspace/work_dirs \
    -v $(pwd)/runs:/workspace/runs \
    -v $(pwd)/logs:/workspace/logs \
    -e NUM_CLASSES=3 \
    -e CONFIG=configs/pidnet/pidnet-s_2xb6-120k_1024x1024-cityscapes.py \
    -e RUN_NAME=pidnet-s-mydata \
    pidnet
Переменные окружения
Переменная	Обязательна	Дефолт	Описание
NUM_CLASSES	нет	3	Число классов
CONFIG	нет	configs/pidnet/pidnet-s_2xb6-120k_1024x1024-cityscapes.py	Конфиг MMSeg
RUN_NAME	нет	pidnet-s	Имя запуска
Остальные переменные — как у DeepLabV3+.

Mask2Former
Репозиторий: MMSegmentation v0.30.0 + mmdet
Архитектура: Masked-attention Mask Transformer.
Лицензия: Apache 2.0.

Что это даёт
Mask2Former — универсальная модель: одна архитектура для semantic + instance + panoptic сегментации. Masked-attention в transformer decoder позволяет фокусироваться на предсказанных масках, а не на всём изображении.

Результаты (Cityscapes):

R-50: 80.44 mIoU

Swin-L: 83.52 mIoU

Устойчива к изменениям контраста (в отличие от U-Net и SegFormer) — важное свойство для задач с переменным освещением.

⚠️ Требует mmdet
Mask2Former падает без MMDetection с TypeError: Mask2FormerHead ... unexpected keyword argument 'in_channels'. В контейнере mmdet==2.27.0 уже установлен.

Сборка и запуск
bash
docker build -f mask2former_docker.dockerfile -t mask2former .

docker run -it --rm --gpus all --shm-size=8g \
    -v $(pwd)/datasets:/workspace/datasets \
    -v $(pwd)/work_dirs:/workspace/work_dirs \
    -v $(pwd)/runs:/workspace/runs \
    -v $(pwd)/logs:/workspace/logs \
    -e NUM_CLASSES=3 \
    -e CONFIG=configs/mask2former/mask2former_r50_8xb2-90k_cityscapes-512x1024.py \
    -e RUN_NAME=mask2former-mydata \
    mask2former
Переменные окружения
Переменная	Обязательна	Дефолт	Описание
NUM_CLASSES	нет	3	Число классов
CONFIG	нет	configs/mask2former/mask2former_r50_8xb2-90k_cityscapes-512x1024.py	Конфиг MMSeg
RUN_NAME	нет	mask2former	Имя запуска
Остальные переменные — как у DeepLabV3+.

Detectron2 Mask R-CNN
Репозиторий: Detectron2 v0.6
Архитектура: Faster R-CNN + mask head (FCN на RoIAlign).
Лицензия: Apache 2.0.

Что это даёт
Mask R-CNN — это instance segmentation. В отличие от semantic, модель различает отдельные экземпляры объектов. Два человека на фото — это «человек #1» и «человек #2», каждый со своей маской.

Метрика: mask AP (Average Precision), а не mIoU.

Формат данных: COCO с полем segmentation (полигоны или RLE).

Сборка и запуск
bash
docker build -f detectron2_maskrcnn_docker.dockerfile -t mask-rcnn .

docker run -it --rm --gpus all --shm-size=8g \
    -v $(pwd)/datasets:/workspace/datasets \
    -v $(pwd)/output:/workspace/output \
    -v $(pwd)/runs:/workspace/runs \
    -v $(pwd)/logs:/workspace/logs \
    -e NUM_CLASSES=3 \
    -e BATCH=2 \
    -e EPOCHS=12 \
    mask-rcnn
Переменные окружения
Переменная	Обязательна	Дефолт	Описание
NUM_CLASSES	нет	3	Число классов
BATCH	нет	2	Batch size
EPOCHS	нет	12	Число эпох
MAX_ITER	нет	30000	Число итераций
MODEL_CFG	нет	COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml	Конфиг из Model Zoo
OUTPUT_DIR	нет	/workspace/output	Куда писать чекпоинты
COCO_DIR	нет	/workspace/datasets/coco	COCO-датасет
MONAI
Репозиторий: Project-MONAI/MONAI
Базовый образ: nvcr.io/nvidia/pytorch:24.08-py3
Лицензия: Apache 2.0.

Что это даёт
MONAI — специализированный фреймворк для медицинской визуализации:

3D-данные (DICOM, NIfTI) — нативные форматы, не JPG/PNG.

3D-архитектуры: UNETR, SwinUNETR, SegResNet.

Метрики: Dice, Hausdorff Distance.

Трансформы для медицинских данных: RandSpatialCropd, ScaleIntensityd, LoadImaged.

⚠️ Не для промышленного CV
MONAI работает с 3D-объёмами и медицинскими форматами. Для 2D-изображений на промышленных задачах используй MMSegmentation (DeepLabV3+, U-Net, SegFormer, PIDNet, Mask2Former).

MONAI включается в README для полноты картины — если у тебя есть медицинские проекты.

Сборка и запуск
bash
docker build -f monai_docker.dockerfile -t monai .

docker run --gpus all --ipc host --net host monai
Особенности запуска
--ipc host — MONAI активно использует shared memory для 3D-данных.

--net host — если нужна интеграция с DICOM-сервером.

Внутри контейнера — nvcr.io/nvidia/pytorch:24.08-py3 с CUDA 12.x.

Какую модель выбирать
По задаче
Задача	Рекомендация
Real-time semantic segmentation (≥30 FPS)	PIDNet-S/M
Максимальная точность semantic	Mask2Former (Swin-L), SegFormer
Чёткиe границы объектов	DeepLabV3+
Малые данные, медицина	U-Net
Instance segmentation	Detectron2 Mask R-CNN
Медицинская 3D-сегментация	MONAI
Универсальность (semantic+instance)	Mask2Former
Сложные сцены с переменным освещением	Mask2Former
По типу
Semantic (DeepLabV3+, U-Net, SegFormer, PIDNet, Mask2Former):

Классифицируют каждый пиксель.

Не различают экземпляры.

Метрика: mIoU.

Instance (Mask R-CNN):

Различают экземпляры.

Метрика: mask AP.

Формат данных: COCO с полигонами.

Medical (MONAI):

3D-данные.

Метрика: Dice.

По лицензии
Все семь моделей — Apache 2.0. Коммерческое использование без раскрытия кода разрешено.

Описание блоков кода
Все контейнеры (кроме MONAI) следуют единой структуре.

00_convert.py — конвертация датасета
Что делает: читает LabelMe JSON из datasets/, генерирует PNG-маски (для semantic) или COCO JSON с полигонами (для instance).

Ключевые функции:

find_subfolders(root) — рекурсивно ищет папки с .json, разделяет на trainval и test по имени.

make_masks(json_files, label_list, img_dir, ann_dir) — создаёт PNG-маски через cv2.fillPoly.

main() — обходит все JSON, копирует JPG, пишет маски, делает train/val split.

Переменные env: DATASETS_DIR, OBJ_NAMES_FILE, SPLIT_SIZE, RANDOM_STATE, MMSEG_DATA_DIR.

01_train.sh — тренировка
DeepLabV3+, U-Net, SegFormer, PIDNet, Mask2Former: вызывают tools/train.py из MMSegmentation с конфигом из env.

Detectron2 Mask R-CNN: использует DefaultTrainer из detectron2.engine.

Все пишут лог в logs/01_train.log через tee -a.

02_test_pth.sh — тест .pth
Что делает: загружает обученный .pth, прогоняет тестовый набор, считает mIoU через MMSeg eval.

Почему это baseline: с этим результатом сравнивается ONNX-версия. Если mIoU .pth и .onnx отличаются >1% — проблема в экспорте.

03_export_onnx.sh — экспорт в ONNX
MMSeg-контейнеры: tools/deployment/pytorch2onnx.py с --verify.
Detectron2: torch.onnx.export() с wrapper'ом для tensor-выхода.

04_test_onnx.py — тест .onnx
Что делает: загружает ONNX через onnxruntime, прогоняет тестовый набор, считает mIoU.

Ключевые функции:

preprocess(img, imgsz) — ресайз + нормализация (mean/std из MMSeg).

compute_miou(preds, gts, num_classes) — считает IoU по каждому классу, усредняет.

main() — обход тестового набора, сбор ious, средний mIoU.

На выходе: logs/04_test_onnx.log с mIoU и avg latency.

pipeline.sh — главный entrypoint
Что делает:

Пишет в мастер-лог 00_pipeline.log каждый шаг с таймстампом.

Последовательно вызывает все стадии.

При ошибке фиксирует exit code и останавливается (set -e + trap).

Мастер-лог — первый файл при проблеме:

text
[2026-03-30 14:23:11] [PIPELINE] STEP 1/6: Training started → logs/01_train.log
[2026-03-30 16:45:33] [PIPELINE] STEP 1/6: Training finished in 8532s
[2026-03-30 16:47:12] [PIPELINE] STEP 3/6: Export finished in 45s
При падении — увидишь FAILED at step X/6 (exit=N).

Что на выходе
Модели
text
work_dirs/<RUN_NAME>/
├── best_mIoU_epoch_*.pth
├── latest.pth
└── ...

runs/onnx/model.onnx
Для Detectron2 чекпоинты лежат в output/.

Логи
text
logs/
├── 00_pipeline.log      # мастер-лог: все шаги с таймстампами и exit code
├── 00_convert.log       # конвертация датасета
├── 01_train.log         # тренировка (loss, mIoU по эпохам)
├── 02_test_pth.log      # mIoU .pth
├── 03_export.log        # экспорт в ONNX
├── 04_test_onnx.log     # mIoU .onnx
└── plots/
    ├── 01_training_curves.png
    ├── 02_pr_curves.png
    └── ...
Как читать логи
Первый файл при проблеме — 00_pipeline.log. Там видно, на каком шаге упало.

Сравнение .pth vs .onnx:

bash
grep "mIoU" logs/02_test_pth.log logs/04_test_onnx.log
Если разница >1% — проблема в экспорте.

Диагностика проблем
Bus error при тренировке
Мало shared memory. Увеличь --shm-size до 16g или 32g.

ImportError: libGL.so.1 или libgthread-2.0.so.0
Не хватает системных библиотек для OpenCV. Проверь, что базовый образ не подменён на -runtime вместо -devel.

CUDA out of memory
Уменьши BATCH. Ориентиры:

DeepLabV3+ (R-101) при 512×512 — BATCH=2

U-Net (S5-D16) — BATCH=4

SegFormer (MiT-B2) — BATCH=2

PIDNet-S — BATCH=4

Mask2Former (R-50) — BATCH=2

Detectron2 Mask R-CNN — BATCH=2

ONNX mIoU сильно ниже, чем .pth
Три частые причины:

Opset version. Попробуй opset=13 вместо 16/17.

Execution provider. Проверь, что ONNX Runtime использует GPU:

python
print(session.get_providers())  # CUDAExecutionProvider должен быть первым
Preprocessing. Убедись, что в 04_test_onnx.py тот же ресайз, нормализация и порядок каналов, что и в .pth-тесте.

Mask2Former: TypeError: Mask2FormerHead ... unexpected keyword argument 'in_channels'
Не установлен MMDetection. В контейнере mmdet==2.27.0 уже есть. Если ошибка сохраняется — проверь, что pip install mmdet==2.27.0 выполнился в Dockerfile.

Mask2Former: конфликт mmcv версий
MMDetection и MMSegmentation используют один и тот же mmcv-full 1.7.0. Если при установке mmdet перезаписал mmcv — переустанови mmcv-full==1.7.0.

MONAI: недостаточно shared memory для 3D
MONAI активно использует shared memory для 3D-данных. Запускай с --ipc host, а не с --shm-size.

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
Подменили базовый образ на другую версию PyTorch. Проверь FROM в Dockerfile — для первых шести только pytorch/pytorch:1.10.0-cuda11.3-cudnn8-devel.

Лицензии
Модель					Лицензия кода	Лицензия весов	Коммерческое использование
DeepLabV3+				Apache 2.0		Apache 2.0	✅ 		 Да
U-Net					Apache 2.0		Apache 2.0	✅ 		 Да
SegFormer				Apache 2.0		Apache 2.0	✅ 		 Да
PIDNet					Apache 2.0		Apache 2.0	✅		 Да
Mask2Former				Apache 2.0		Apache 2.0	✅		 Да
Detectron2 Mask R-CNN	Apache 2.0		Apache 2.0	✅		 Да
MONAI					Apache 2.0		Apache 2.0	✅ 		 Да
Все семь контейнеров можно использовать в закрытых коммерческих продуктах без юридических рисков.