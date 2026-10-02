CV Group 3: Depth, Restoration, Face, KIE, Video, Generative
Двенадцать Docker-контейнеров для depth estimation, restoration, face recognition, KIE/layout, video understanding и генерации изображений.

Обзор
#	Модель	Задача	Лицензия	Особенность
1	Depth Anything V2	Depth estimation	Apache 2.0	SOTA 2024, 25M–335M params
2	MiDaS	Depth estimation	MIT	Intel ISL, лёгкий
3	Real-ESRGAN	Restoration	Apache 2.0	Blind super-resolution
4	MMagic	Restoration	Apache 2.0	Полный тулкит
5	UniFace	Face recognition	MIT	All-in-one: detection + recognition + landmarks
6	RetinaFace	Face detection	MIT	Только детекция + landmarks
7	LayoutParser	Layout analysis	Apache 2.0	Структура документов
8	Table Transformer	Table recognition	MIT	Microsoft TATR
9	MMAction2	Video understanding	Apache 2.0	OpenMMLab тулкит
10	VideoMAE	Video understanding	Apache 2.0	Self-supervised
11	FLUX.2 [klein] 4B	Text-to-image + editing	Apache 2.0	Sub-second inference, 8.4 GB VRAM
12	Stable Diffusion XL	Text-to-image	CreativeML Open RAIL++-M	Максимальная кастомизация
Face Recognition: почему UniFace, а не InsightFace
Проблема с InsightFace:

Код — MIT, но предобученные модели — non-commercial (тренировочные данные имеют ограничения).

Решение — UniFace:

MIT-лицензия на весь код, включая веса AdaFace/ArcFace/MobileFace.

All-in-one пайплайн: детекция (RetinaFace, SCRFD, YOLOv5Face, YOLOv8Face) + распознавание (AdaFace, ArcFace, MobileFace) + 106-point landmarks + атрибуты (возраст, пол, эмоции).

ONNX Runtime — автоматическое ускорение на CPU/GPU/Apple Silicon.

Готов к production: lightweight, easy to integrate.

Альтернатива SFace:

Apache 2.0, commercial-permissive.

Но: только recognition (эмбеддинги 128-D), нужен отдельный детектор лиц.

Лёгкая (~37MB), быстрая (8.65 ms на Intel CPU).

Подходит, если у тебя уже есть свой детектор.

Итог: Для полного commercial-clean пайплайна (детекция + распознавание + landmarks) — UniFace. Если нужен только recognition и есть свой детектор — SFace.

UniFace
Лицензия: MIT.

Что это: all-in-one face analysis toolkit: детекция, распознавание, landmarks, возраст/пол/эмоции, anti-spoofing, трекинг, парсинг.

Компоненты:

Детекторы: RetinaFace, SCRFD, YOLOv5Face, YOLOv8Face.

Распознавание: AdaFace, ArcFace, MobileFace, SphereFace.

Landmarks: 106-point.

Атрибуты: возраст, пол, эмоции.

Docker:

bash
docker build -f uniface_docker.dockerfile -t uniface .

# Детекция + распознавание
docker run ... uniface python /workspace/inference.py --input /workspace/images

# Верификация (сравнение двух лиц)
docker run ... uniface python /workspace/verify.py --img1 a.jpg --img2 b.jpg
Когда выбирать: полный face-пайплайн для коммерческого продукта.

RetinaFace
Лицензия: MIT.

Что это: single-stage face detector с 5 landmarks. Только детекция, без recognition.

Docker:

bash
docker build -f retinaface_docker.dockerfile -t retinaface .
docker run ... retinaface python /workspace/inference.py --input /workspace/images
Когда выбирать: если нужна только детекция лиц (без распознавания).

Depth Anything V2
Лицензия: Apache 2.0.

Что это: SOTA monocular depth estimation 2024. DINOv2 encoder + DPT head. Нативный вход 518×518.

Варианты:

Small: ~25M params, ~40 fps 1080p на 3090.

Base: ~97M params, ~20 fps.

Large: ~335M params, ~8 fps, sharpest edges.

Metric fine-tunes: Indoor (Hypersim, до 20м), Outdoor (Virtual KITTI, до 80м).

Docker:

bash
docker build -f depth_anything_docker.dockerfile -t depth-anything .
docker run ... -e DEPTH_SIZE=small depth-anything python /workspace/inference.py --input /workspace/images
MiDaS
Лицензия: MIT.

Что это: классический depth estimator от Intel ISL. DPT head + Swin/BEiT backbone.

Варианты:

dpt-swinv2-tiny-256: ~60 fps 1080p, минимум VRAM.

dpt-beit-large-512: ~6 fps, best quality.

Docker:

bash
docker build -f midas_docker.dockerfile -t midas .
docker run ... -e DEPTH_SIZE=tiny midas python /workspace/inference.py --input /workspace/images
Real-ESRGAN
Лицензия: Apache 2.0 (MMagic).

Что это: blind super-resolution для real-world изображений. Обучен на синтетических деградациях. RRDB generator + U-Net discriminator.

Docker:

bash
docker build -f real_esrgan_docker.dockerfile -t real-esrgan .
docker run ... -e INPUT_DIR=/workspace/images -e OUTPUT_DIR=/workspace/runs real-esrgan /workspace/inference.sh
MMagic
Лицензия: Apache 2.0.

Что это: OpenMMLab тулкит для restoration, generation, matting. Поддерживает Real-ESRGAN, ESRGAN, BasicVSR, StyleGAN и др.

Docker:

bash
docker build -f mmagic_docker.dockerfile -t mmagic .
docker run ... -e CONFIG=... -e CHECKPOINT=... mmagic /workspace/inference.sh
LayoutParser
Лицензия: Apache 2.0.

Что это: документ layout analysis. Определяет блоки: Text, Title, List, Table, Figure.

Docker:

bash
docker build -f layoutparser_docker.dockerfile -t layoutparser .
docker run ... layoutparser python /workspace/inference.py --input /workspace/images
Table Transformer
Лицензия: MIT.

Что это: Microsoft TATR — detection + structure recognition таблиц.

Docker:

bash
docker build -f table_transformer_docker.dockerfile -t table-transformer .
docker run ... table-transformer python /workspace/inference.py --input /workspace/images
MMAction2
Лицензия: Apache 2.0.

Что это: OpenMMLab тулкит для video understanding: action recognition, temporal detection, spatio-temporal detection.

Модели: VideoMAE, MViT V2, UniFormer, TSN, TSM, SlowFast.

Docker:

bash
docker build -f mmaction2_docker.dockerfile -t mmaction2 .
docker run ... mmaction2 /workspace/inference.sh
VideoMAE
Лицензия: Apache 2.0.

Что это: masked autoencoder для видео. Входит в MMAction2.

Docker:

bash
docker build -f videomae_docker.dockerfile -t videomae .
docker run ... videomae /workspace/inference.sh
FLUX.2 [klein] 4B
Лицензия: Apache 2.0 — коммерчески чистая, включая выходные изображения.

Что это: самая быстрая и компактная модель из семейства FLUX. 4B параметров, rectified flow transformer. Генерация + редактирование в одной архитектуре.

Ключевые цифры:

Sub-second inference (~0.3s на GB200, ~1.2s на RTX 5090).

VRAM: 8.4 GB — работает на RTX 3090/4070.

Полная модель: ~13 GB VRAM.

Docker:

bash
docker build -f flux_docker.dockerfile -t flux-klein .

# Text-to-image (distilled, 4 steps)
docker run -it --rm --gpus all --shm-size=8g \
    -v $(pwd)/runs:/workspace/runs \
    -v $(pwd)/logs:/workspace/logs \
    flux-klein python /workspace/generate.py \
        --prompt "Industrial steel surface with defect, macro photography" \
        --steps 4 --guidance 1.0 \
        --output /workspace/runs/flux_defect.png

# Image editing
docker run -it --rm --gpus all --shm-size=8g \
    -v $(pwd)/images:/workspace/images \
    -v $(pwd)/runs:/workspace/runs \
    flux-klein python /workspace/edit.py \
        --prompt "Remove the defect, restore clean metal surface" \
        --image /workspace/images/defect.jpg
Когда выбирать:

Быстрая генерация + редактирование.

Коммерческий продукт (Apache 2.0).

Sub-second latency.

Image-to-image задачи (удаление дефектов, inpainting).

Stable Diffusion XL
Лицензия: CreativeML Open RAIL++-M — коммерчески разрешена с ограничениями (запрещено генерировать незаконный/вредоносный контент).

Что это: самая кастомизируемая open-source модель. LoRA, ControlNet, Inpainting, тысячи community-моделей. SDXL base + refiner ensemble даёт значительно лучшее качество, чем SD 1.5/2.1.

Docker:

bash
docker build -f stable_diffusion_docker.dockerfile -t sdxl .

docker run -it --rm --gpus all --shm-size=8g \
    -v $(pwd)/runs:/workspace/runs \
    -v $(pwd)/logs:/workspace/logs \
    sdxl python /workspace/generate.py \
        --prompt "Industrial defect on steel surface, macro, high detail" \
        --steps 40 \
        --output /workspace/runs/sdxl_defect.png
Когда выбирать:

Кастомизация (LoRA/ControlNet).

Эксперименты с промптами.

Community-модели (fine-tunes под конкретные домены).

Когда нужен ControlNet для точного контроля композиции.

Какую модель выбирать
Задача	Рекомендация
Depth: SOTA качество	Depth Anything V2 Large
Depth: real-time	Depth Anything V2 Small
Depth: минимум VRAM	MiDaS tiny
Restoration: real-world	Real-ESRGAN
Restoration: полный тулкит	MMagic
Face: полный пайплайн (коммерция)	UniFace
Face: только детекция	RetinaFace
Face: только recognition (свой детектор)	SFace (OpenCV Zoo)
Layout analysis	LayoutParser
Table recognition	Table Transformer
Video: OpenMMLab	MMAction2
Video: self-supervised	VideoMAE
Text-to-image: коммерция + скорость	FLUX.2 [klein] 4B
Text-to-image: кастомизация	Stable Diffusion XL
Image editing	FLUX.2 [klein] 4B
Аугментация через генерацию	FLUX.2 [klein] 4B или SDXL

Лицензии (сводка)
Модель					Лицензия		Коммерческое использование
Depth Anything V2		Apache 2.0				✅ Да
MiDaS						MIT					✅ Да
Real-ESRGAN				Apache 2.0				✅ Да
MMagic					Apache 2.0				✅ Да
UniFace						MIT					✅ Да
RetinaFace					MIT					✅ Да
SFace (OpenCV Zoo)		Apache 2.0				✅ Да
LayoutParser			Apache 2.0				✅ Да
Table Transformer			MIT					✅ Да
MMAction2				Apache 2.0				✅ Да
VideoMAE				Apache 2.0				✅ Да
FLUX.2 [klein] 4B		Apache 2.0				✅ Да
Stable Diffusion XL	CreativeML Open RAIL++-M	✅ Да (с ограничениями)
Все модели в этой группе — commercial-clean (с учётом ограничений Stable Diffusion на вредоносный контент).