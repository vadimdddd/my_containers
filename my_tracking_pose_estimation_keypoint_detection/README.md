# CV Tracking, Pose & Keypoint Pipeline

Шесть Docker-контейнеров для трекинга, оценки позы и keypoint detection.

---

## Обзор

| # | Модель | Задача | Лицензия | Особенность |
|---|---|---|---|---|
| 1 | **ByteTrack** | MOT (трекинг) | MIT | SOTA трекер, работает поверх YOLO |
| 2 | **MMTracking** | MOT/SOT/VID | Apache 2.0 | Модульный конструктор |
| 3 | **MMPose** | Pose estimation | Apache 2.0 | 2D body/face/hand/animal |
| 4 | **RTMPose** | Pose (real-time) | Apache 2.0 | SOTA real-time, ONNX/TensorRT |
| 5 | **MMPose Keypoint** | Custom keypoints | Apache 2.0 | Любые точки на объектах |
| 6 | **Detectron2 Keypoint** | Keypoint R-CNN | Apache 2.0 | One category keypoints |

---

## ByteTrack

**Лицензия:** MIT.

**Что это:** не обучаемая модель, а **алгоритм трекинга**. Работает поверх готовых детекций (от YOLO). Связывает объекты во времени через ассоциацию по двум порогам confidence [citation:38].

**Docker:**
```bash
docker build -f bytetrack_docker.dockerfile -t bytetrack .
docker run ... bytetrack python /workspace/track.py --video input.mp4
Когда выбирать: подсчёт объектов, анализ трафика, поведенческая аналитика.

MMTracking
Лицензия: Apache 2.0.

Что это: модульный трекер от OpenMMLab. MOT (SORT, DeepSORT, ByteTrack, QDTrack), SOT (SiameseRPN, STARK), VID .

Docker:

bash
docker build -f mmtracking_docker.dockerfile -t mmtracking .
docker run ... -e CONFIG=configs/mot/deepsort/... mmtracking
Когда выбирать: если ты уже в OpenMMLab, нужна модульность.

MMPose
Лицензия: Apache 2.0.

Что это: модульный pose estimation. 2D body (COCO 17 keypoints), face, hand, animal .

Ключевые цифры:

Swin-T: 72.4 AP (COCO) 

HRNet-W48: 75.1 AP (COCO) 

Docker:

bash
docker build -f mmpose_docker.dockerfile -t mmpose .
docker run ... -e NUM_KEYPOINTS=17 mmpose
Когда выбирать: если нужна модульность и поддержка разных keypoint конфигураций.

RTMPose
Лицензия: Apache 2.0.

Что это: real-time multi-person pose estimation. CSPNeXt backbone + SimCC head .

Ключевые цифры (Body8):

RTMPose-t: 65.9 AP 

RTMPose-m: 74.9 AP 

RTMPose-l: 76.7 AP 

Docker:

bash
docker build -f rtmpose_docker.dockerfile -t rtmpose .
docker run ... rtmpose python /workspace/inference.py --video input.mp4
Когда выбирать: real-time pose, edge deployment.

MMPose Keypoint (custom)
Лицензия: Apache 2.0.

Что это: MMPose с кастомным датасетом. Для любых точек на объектах (не только human pose) .

Ключевое отличие: keypoint_head.out_channels = NUM_KEYPOINTS, dataset_info.keypoint_num = NUM_KEYPOINTS.

Docker:

bash
docker build -f mmpose_keypoint_docker.dockerfile -t mmpose-keypoint .
docker run ... -e NUM_KEYPOINTS=4 -e KEYPOINT_NAMES="corner_tl corner_tr corner_bl corner_br" mmpose-keypoint
Когда выбирать: промышленная метрология, робототехника (захват), аугментация.

Detectron2 Keypoint R-CNN
Лицензия: Apache 2.0.

Что это: Faster R-CNN + KeypointHead. Для keypoint detection .

⚠️ Ограничение: Detectron2 поддерживает только одну категорию для keypoints. Все keypoints принадлежат одному классу объектов .

Docker:

bash
docker build -f detectron2_keypoint_docker.dockerfile -t detectron2-kpts .
docker run ... -e NUM_KEYPOINTS=4 detectron2-kpts
Когда выбирать: если нужен один класс объектов с keypoints.

Какую модель выбирать
Задача	Рекомендация
Трекинг: production	ByteTrack
Трекинг: OpenMMLab	MMTracking
Pose: модульность	MMPose
Pose: real-time	RTMPose
Custom keypoints (multi-class)	MMPose Keypoint
Custom keypoints (one-class)	Detectron2 Keypoint
Лицензии
Все шесть — MIT или Apache 2.0. Коммерческое использование разрешено.