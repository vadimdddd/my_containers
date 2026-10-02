# CV OCR Pipeline: LightOnOCR / PaddleOCR / MMOCR

Три Docker-контейнера для задач распознавания текста с **permissive-лицензиями**.

**Важное деление:**
- **LightOnOCR-2-1B** — end-to-end VLM. На вход картинка, на выход текст.
- **PaddleOCR** — традиционный пайплайн (детектор + распознаватель).
- **MMOCR** — модульный конструктор (как MMDetection/MMSegmentation).

---

## Содержание

- [Обзор моделей](#обзор-моделей)
- [LightOnOCR-2-1B](#lightonocr-2-1b)
- [PaddleOCR](#paddleocr)
- [MMOCR](#mmocr)
- [Какую модель выбирать](#какую-модель-выбирать)
- [Лицензии](#лицензии)

---

## Обзор моделей

| Модель | Тип | Лицензия | Размер | Подход |
|---|---|---|---|---|
| **LightOnOCR-2-1B** | VLM | Apache 2.0 | 1B | End-to-end |
| **PaddleOCR** | Toolkit | Apache 2.0 | ~100MB | Детектор + распознаватель |
| **MMOCR** | Toolkit | Apache 2.0 | ~100MB | Модульный конструктор |

---

## LightOnOCR-2-1B

**Лицензия:** Apache 2.0 [citation:1][citation:2][citation:7].

**Что это:** 1B-параметровая end-to-end vision-language модель. Конвертирует изображения документов (PDF, сканы) в чистый текст без хрупких OCR-пайплайнов.

**Ключевые цифры:**
- SOTA на OlmOCR-Bench при размере ~9× меньше конкурентов [citation:7].
- 5.71 страниц/сек на H100 (~493k страниц/день) за <$0.01 за 1000 страниц.
- Скорость: 3.3× быстрее Chandra OCR, 1.7× быстрее OlmOCR, 5× быстрее dots.ocr [citation:7].

**Варианты:** `LightOnOCR-2-1B` (лучший OCR), `-bbox` (bounding boxes), `-base` (для файн-тюнинга) [citation:1].

**Файн-тюнинг:** LoRA/QLoRA. Требует 24GB+ VRAM для full-precision, ~8GB для 4-bit [citation:2][citation:8].

**Когда выбирать:**
- Документы, сканы, таблицы, чеки, формулы.
- Мультиязычные документы.
- Если нужен end-to-end (без пайплайна детектор+распознаватель).

**Docker:**
```bash
docker build -f lightonocr_docker.dockerfile -t lightonocr .

# Инференс
docker run -it --rm --gpus all --shm-size=8g \
    -v $(pwd)/images:/workspace/images \
    -v $(pwd)/logs:/workspace/logs \
    lightonocr python /workspace/inference.py --image /workspace/images/doc.png

# LoRA файн-тюнинг
docker run -it --rm --gpus all --shm-size=8g \
    -v $(pwd)/datasets:/workspace/datasets \
    -v $(pwd)/runs:/workspace/runs \
    lightonocr python /workspace/finetune_lora.py --data /workspace/datasets
PaddleOCR
Лицензия: Apache 2.0.

Что это: тулкит для традиционного OCR. Детектор текста (DB) + распознаватель (CRNN/SVTR).

Плюсы:

Мультиязычность (80+ языков).

ONNX/TensorRT экспорт.

Готовая экосистема.

Минусы:

Документация в основном на китайском.

Официальные Docker-образы:

paddleocr-vl:latest-nvidia-gpu (~13 GB) .

paddleocr-vl:latest-amd-gpu (~15 GB) .

Когда выбирать:

Промышленный OCR с готовыми пайплайнами.

Если нужен ONNX/TensorRT.

Docker:

bash
docker build -f paddleocr_docker.dockerfile -t paddleocr .

docker run -it --rm --gpus all --shm-size=8g \
    -v $(pwd)/datasets:/workspace/datasets \
    -v $(pwd)/logs:/workspace/logs \
    -e TASK=det \
    paddleocr
MMOCR
Лицензия: Apache 2.0.

Что это: модульный OCR-конструктор от OpenMMLab. Тот же принцип, что у MMDetection/MMSegmentation.

Ключевые компоненты:

Детекция текста: DBNet, PSENet, PANet.

Распознавание: CRNN, SAR, SEG, ABINet.

KIE (Key Information Extraction).

Метрики:

WordMetric — word-level accuracy (exact, ignore_case, ignore_case_symbol) .

OneMinusNEDMetric — 1-N.E.D (normalized edit distance) .

Пример из твоего workflow:

0_word_acc: 0.958, 0_1-N.E.D: 0.9787 — отличные результаты.

Когда выбирать:

Если ты уже в экосистеме OpenMMLab.

Если нужна модульность (backbone + neck + head).

Если нужны специфические распознаватели (SAR, SEG).

Docker:

bash
docker build -f mmocr_docker.dockerfile -t mmocr .

docker run -it --rm --gpus all --shm-size=8g \
    -v $(pwd)/datasets:/workspace/datasets \
    -v $(pwd)/work_dirs:/workspace/work_dirs \
    -e CONFIG=configs/textrecog/crnn/crnn_nlmk_chvk_dataset.py \
    mmocr
Какую модель выбирать
Задача	Рекомендация
End-to-end OCR документов	LightOnOCR-2-1B
Промышленный OCR с ONNX	PaddleOCR
Модульность, CRNN, SAR, SEG	MMOCR
Мультиязычные документы	LightOnOCR или PaddleOCR
Файн-тюнинг на кастомные данные	MMOCR (проще), LightOnOCR (LoRA)
Мелкий текст, строки	MMOCR CRNN
Лицензии
Модель	Лицензия	Коммерческое использование
LightOnOCR-2-1B	Apache 2.0	✅ Да
PaddleOCR	Apache 2.0	✅ Да
MMOCR	Apache 2.0	✅ Да
Все три контейнера можно использовать в закрытых коммерческих продуктах.