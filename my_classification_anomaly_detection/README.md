CV Classification & Anomaly Detection & Generation Pipeline
Пять Docker-контейнеров для задач классификации, поиска аномалий и генерации изображений.

Обзор
#	Модель	Задача	Лицензия	Особенность
1	timm	Классификация	Apache 2.0	800+ моделей, простой API
2	MMPretrain	Классификация	Apache 2.0	Модульный конструктор
3	Anomalib	Anomaly detection	Apache 2.0	15+ алгоритмов, PatchCore
4	PatchCore	Anomaly detection	Apache 2.0	Standalone, WideResNet-50
5	BigGAN	Class-conditional генерация	MIT	FID 36.94, ImageNet
timm
Лицензия: Apache 2.0.

Что это: библиотека с 800+ предобученными моделями для классификации. timm.create_model('resnet50', pretrained=True) — и модель готова [citation:13].

Docker:

bash
docker build -f timm_docker.dockerfile -t timm-cls .
docker run -it --rm --gpus all --shm-size=8g \
    -v $(pwd)/datasets:/workspace/datasets \
    -v $(pwd)/runs:/workspace/runs \
    -e MODEL_NAME=resnet50 -e EPOCHS=20 \
    timm-cls
Когда выбирать: быстрый прототип, одна модель, гибкость.

MMPretrain
Лицензия: Apache 2.0.

Что это: классификация от OpenMMLab. Fine-tuning через config inheritance: копируешь базовый конфиг, меняешь num_classes и frozen_stages.

Docker:

bash
docker build -f mmpretrain_docker.dockerfile -t mmpretrain .
docker run ... -e CONFIG=configs/resnet/resnet50_8xb32_in1k.py mmpretrain
Когда выбирать: если ты уже в OpenMMLab, нужна модульность.

Anomalib
Лицензия: Apache 2.0.

Что это: Intel-фреймворк для anomaly detection. Обучение только на нормальных данных, поиск дефектов без разметки.

Ключевые цифры:

PatchCore: 0.980 AUROC на MVTec AD (image-level).

Robust к domain shift.

Docker:

bash
docker build -f anomalib_docker.dockerfile -t anomalib .
docker run ... -e ANOMALIB_MODEL=Patchcore -e CATEGORY=default anomalib
Когда выбирать: промышленный контроль качества, дефекты редки, разметка сложна.

PatchCore (standalone)
Лицензия: Apache 2.0.

Что это: чистая реализация PatchCore без Anomalib. Полезно для кастомизации.

Алгоритм:

Pretrained CNN (WideResNet-50) извлекает mid-level features.

Memory bank из patch-level features.

Coreset subsampling для ускорения.

Anomaly score = max distance до nearest neighbour.

Docker:

bash
docker build -f patchcore_docker.dockerfile -t patchcore .
docker run ... patchcore
Когда выбирать: нужна кастомизация алгоритма.

BigGAN
Лицензия: MIT.

Что это: class-conditional GAN 2018 года от DeepMind. Управляешь классом объекта через embedding — модель генерирует изображения заданного класса.

Ключевые цифры (ImageNet):

FID 36.94 (против 67.82 у DCGAN) — почти в 2 раза лучше.

IS 7.61 (против 4.87).

SSIM 0.856 (против 0.731).

PSNR 28.12 (против 22.15).

Варианты весов: ImageNet 128×128, 256×256, 512×512; CIFAR-10 32×32. Доступны через MMagic и HuggingFace.

Docker:

bash
docker build -f biggan_docker.dockerfile -t biggan .

docker run -it --rm --gpus all --shm-size=8g \
    -v $(pwd)/runs:/workspace/runs \
    -v $(pwd)/logs:/workspace/logs \
    -e CLASS_ID=207 \
    biggan python /workspace/generate.py --class-id 207 --num-images 4
Когда выбирать:

Class-conditional генерация.

Эксперименты с GAN.

Baseline для сравнения с diffusion-моделями.

Аугментация данных (генерация дополнительных примеров класса).

⚠️ BigGAN vs DCGAN: BigGAN даёт значительно лучшее качество (FID 36.94 против 67.82), более разнообразные изображения и лучше работает на сложных классах. DCGAN стоит использовать только для обучения/экспериментов — для production бери BigGAN.

Какую модель выбирать
Задача	Рекомендация
Классификация: быстрый прототип	timm
Классификация: OpenMMLab	MMPretrain
Anomaly detection: production	Anomalib
Anomaly detection: кастомизация	PatchCore standalone
Class-conditional генерация	BigGAN
Аугментация данных через генерацию	BigGAN
Лицензии
Модель		Лицензия	Коммерческое использование
timm		Apache 2.0		✅ Да
MMPretrain	Apache 2.0		✅ Да
Anomalib	Apache 2.0		✅ Да
PatchCore	Apache 2.0		✅ Да
BigGAN			MIT			✅ Да
Все пять — permissive (MIT или Apache 2.0). Коммерческое использование разрешено без раскрытия кода.