# Перцепт — YOLO инференс на NPU (RK3588S)

Сервер обнаружения объектов с камеры в реальном времени.
Поддерживает **111 моделей** Ultralytics YOLO (v3/v5/v8/v9/v10/v11/v12/v26)
во всех шести задачах: detect, segment, pose, classify, OBB, semantic segmentation.

**Архитектура**: `CameraSource` → `VideoLoop` (capture → Pipeline) → Flask HTTP + WebSocket → браузер.

**Целевое железо**: Orange Pi 5 Pro (RK3588S), NPU 6 TOPS.

## Статус

| Компонент | Состояние |
|-----------|-----------|
| Инференс на NPU (RKNN) | Работает |
| Инференс на CPU (ONNX) | Работает |
| Конвертация YOLO → RKNN | Конвертируются все задачи |
| Eval точности (eval.py) | Для всех 6 задач (det/seg/pose/cls/obb/sem), варианты ONNX/FP16/INT8/INT8-calib |
| FPS бенчмарк | 111 моделей (ONNX/FP16/INT8/INT8-calib) + per-core CPU/NPU |
| Веб-интерфейс | Live viewer + тестовая галерея |
| Камеры: USB/UVC | Работает |
| Камеры: GigE Vision (Basler) | Работает |
| Камеры: RTSP | Работает |
| Камеры: V4L2 | Работает |
| Камеры: RTSP через OpenCV | Работает |
| Камеры: WebRTC | Нестабильно |
| Камеры: GStreamer | Нестабильно |
| Полная русификация кода | ✅ Завершена |
| Стресс-тест NPU/CPU/RAM | ✅ bash + python |
| INT8 калибровка | Автозагрузка датасета (COCO/DOTA/Cityscapes/ImageNet) |

## Требования

- Docker Engine, пользователь в группе `docker`
- `python3` (системный) — для оркестратора бенчмарка и скриптов установки
  датасетов (`benchmark/benchmark.py`, `setup_datasets.py`, `stress_test.py`).
  Тяжёлые зависимости (ultralytics, rknn-toolkit, torch) на хост ставить НЕ нужно —
  всё выполняется в контейнерах.
- RK3588S с библиотеками NPU: `librknnrt.so`, `librknn_api.so`
- USB камера (MJPEG) или GigE Vision камера (Basler)

### Предустановка: RKNN Runtime (версия >= 2.3.2)

Конвертер использует `rknn-toolkit2` v2.3.2 — хостовой runtime должен совпадать.

Проверка текущей версии:
```bash
# Читает строки из бинарника и ищет номер версии
strings /usr/lib/librknnrt.so | grep "librknnrt version"
```

Обновление (требуется root):
```bash
# 1. Остановить rknn_server, если запущен (без ошибки если не запущен)
pgrep -x rknn_server && kill $(pgrep -x rknn_server) || true

# 2. Бэкап текущего runtime перед заменой
sudo cp /usr/lib/librknnrt.so /usr/lib/librknnrt.so.bak

# 3. Скачать свежий librknnrt.so (библиотека NPU runtime)
curl -sL -o /tmp/librknnrt.so \
  "https://raw.githubusercontent.com/airockchip/rknn-toolkit2/master/rknpu2/runtime/Linux/librknn_api/aarch64/librknnrt.so"

# 4. Скачать свежий rknn_server (демон для NPU)
curl -sL -o /tmp/rknn_server \
  "https://raw.githubusercontent.com/airockchip/rknn-toolkit2/master/rknpu2/runtime/Linux/rknn_server/aarch64/usr/bin/rknn_server"

# 5. Заменить библиотеку (требует root, так как файл в /usr/lib)
sudo cp /tmp/librknnrt.so /usr/lib/librknnrt.so

# 6. Заменить бинарник сервера (требует root)
sudo cp /tmp/rknn_server /usr/bin/rknn_server

# 7. Сделать бинарник исполняемым
sudo chmod +x /usr/bin/rknn_server

# 8. Запустить rknn_server в фоне (NPU будет доступна для инференса)
rknn_server &
```

Откат:
```bash
# Остановить текущий rknn_server
pgrep -x rknn_server && kill $(pgrep -x rknn_server) || true

# Восстановить бэкап библиотеки
sudo cp /usr/lib/librknnrt.so.bak /usr/lib/librknnrt.so

# Запустить восстановленный rknn_server
rknn_server &
```

## Быстрый старт

**Перед первым запуском** необходимо сконвертировать модель:

```bash
# FP16 — быстрый старт, без калибровки и скачивания датасетов
./deploy.sh yolo26n-pose
```

Это сконвертирует модель в RKNN (FP16) и запустит сервис.

После этого можно переключать уже сконвертированные модели без повторной конвертации:

```bash
# Переключить на другую модель
curl "http://localhost:5000/reload?model=yolo11n"

# Конвертировать + запустить новую модель
./deploy.sh yolo26n-seg

# Просто перезапустить сервис (без конвертации)
./start.sh
```

Просмотр: `http://<host>:5000/`
Тестовая страница: `http://<host>:5000/test`

Остановка:

```bash
./stop.sh
```

## Автозапуск после перезагрузки

Два компонента должны запускаться автоматически: **Docker** (восстановление контейнеров) и **rknn_server** (NPU runtime).

### Docker

```bash
sudo systemctl enable docker
```

### RKNN Server

Создать юнит `/etc/systemd/system/rknn_server.service`:

```bash
sudo tee /etc/systemd/system/rknn_server.service << 'EOF'
[Unit]
Description=RKNN Server (NPU runtime)
After=network.target

[Service]
Type=simple
ExecStart=/usr/bin/rknn_server
Restart=on-failure
RestartSec=3

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable rknn_server
```

### Проверка

```bash
systemctl is-enabled docker          # должен вывести: enabled
systemctl is-enabled rknn_server     # должен вывести: enabled
pgrep -la rknn_server                # должен вывести: ... /usr/bin/rknn_server
```

### Отключение автозапуска

```bash
sudo systemctl disable docker
sudo systemctl disable rknn_server
sudo rm /etc/systemd/system/rknn_server.service
sudo systemctl daemon-reload
```

## Деплой модели

```bash
# FP16 RKNN — быстрый старт (по умолчанию)
./deploy.sh yolo26n-pose
./deploy.sh yolo26n-pose --fp16          # то же

# ONNX на CPU — без NPU
./deploy.sh yolo26n-pose --onnx

# INT8 — автозагрузка нужного датасета
./deploy.sh yolo11n --int8               # COCO val2017 (~19 GB)
./deploy.sh yolo26n-obb --int8           # DOTAv1 (~1.9 GB)
./deploy.sh yolo26n-sem --int8           # Cityscapes val
./deploy.sh yolo26n-cls --int8           # ImageNet val (~6.3 GB)

# INT8 с калибровкой (точнее, но нужен датасет)
./deploy.sh yolo26n-cls --int8-calib     # ImageNet val (~6.3 GB)

# Конвертация + FPS бенчмарк + запуск
./deploy.sh yolo26n-pose --full-test

# Перезапуск текущей модели из model/
./deploy.sh
```

Имя модели и тип квантизации отображаются в заголовке вкладки браузера и в ответе `/objects`.
`model/quantize.txt` хранит текущий тип: `fp16`, `int8`, `int8-calib`, `onnx`.

## Режимы камеры

| `CAM_MODE` | Описание |
|------------|----------|
| `auto` | GigE → USB → file (автофоллбэк) |
| `ip` | Только GigE по `CAM_SOURCE` |
| `usb` | Только USB камера |
| `file` | Файл/поток из `CAM_SOURCE` |

```bash
CAM_MODE=auto docker compose up -d
CAM_MODE=ip CAM_SOURCE=192.168.5.1 docker compose up -d
```

## Управление моделями

Переключение по имени через `/reload` (без копирования файлов):

```bash
curl "http://localhost:5000/reload?model=yolo26n-sem&backend=cpu"
curl "http://localhost:5000/reload?model=yolo11n-pose"
curl "http://localhost:5000/reload?model=yolo11n-pose&rknn_suffix=_int8"
```

- `?model=` ищет `<name>.rknn`, `<name>.onnx`, `<name>_postprocess.py` в `model/`
- `?backend=cpu` — форсировать ONNX CPU (если RKNN не нужен)
- `?rknn_suffix=_int8` — выбрать INT8 RKNN модель (`_int8`, `_int8-calib`, `_fp16` — суффикс файла в `model/`)
- `?rknn_suffix=` — пустое значение, использовать модель без суффикса
- Без параметров — загружает `model.rknn` / `model.onnx`

## Конвертация новой модели

Скопировать `.pt` в `convert/weights/` и запустить:

```bash
cd convert
docker compose run --rm convert my-model.pt
```

Конвертер (образ `percept-converter:latest` — надстройка над `percept-percept:latest` с
ultralytics/pycocotools/rknn-toolkit2/onnxruntime/onnxslim, работает на aarch64 Orange Pi 5 Pro,
инференс RKNN на реальном NPU через rknnlite):
1. Применяет RKNN-патчи (YOLO26, fuse bug)
2. Экспортирует ONNX + компилирует RKNN
3. Генерирует `postprocess.py` под задачу
4. Копирует результат в `model/`

## Подготовка датасетов

Датасет **скачивается автоматически** при первом запуске `./deploy.sh --int8` (если отсутствует).
Можно загрузить вручную:

```bash
# Все датасеты
python3 setup_datasets.py --all

# Только нужный
python3 setup_datasets.py --coco        # COCO val2017 (~19 GB) — detect, seg, pose
python3 setup_datasets.py --dota        # DOTAv1 (~1.9 GB) — obb
python3 setup_datasets.py --cityscapes  # Cityscapes val — semantic
python3 setup_datasets.py --imagenet    # ImageNet val (~6.3 GB) — classify
```

Для FP16-моделей датасеты **не нужны** — калибровка не выполняется.

## Бенчмарк (4 фазы)

Оркестратор `benchmark/benchmark.py` запускается на хосте (Orange Pi 5 Pro) и сам поднимает
контейнеры (`convert` для convert/validate/eval, `percept` для bench). Четыре независимые
фазы, можно запускать по отдельности:

```bash
cd benchmark

# Фаза 1: конвертация .pt → .onnx + _fp16.rknn + _int8.rknn + _int8-calib.rknn + _postprocess.py
python benchmark.py --phase convert

# Фаза 2: валидация ONNX vs RKNN FP16+INT8 (реальный NPU, fallback — симулятор)
python benchmark.py --phase validate

# Фаза 3: FPS + нагрузка CPU/NPU per-core (ARM железо)
python benchmark.py --phase bench

# Фаза 4: точность (eval.py через Pipeline: top1/top5 для cls,
#          mAP для det/seg/pose/obb, mIoU для sem), все 4 quantize-варианта
python benchmark.py --phase eval
```

### Мульти-фазы

```bash
# Конвертация + валидация
python benchmark.py --phase convert validate

# Всё сразу (convert → validate → bench → eval)
python benchmark.py --phase all

# Только FPS, только INT8
python benchmark.py --phase bench --quantize int8

# Одна модель
python benchmark.py --phase all --model yolo26n

# Только PILOT-модели (по одной nano на линейку)
python benchmark.py --phase all --pilot
```

### Квантизация

| `--quantize` | Описание |
|-------------|----------|
| `onnx` | Только ONNX CPU |
| `none` | Только FP16 RKNN |
| `int8` | INT8 без калибровки (без скачивания датасета) |
| `int8-calib` | INT8 с калибровкой на датасете |
| `all` | Все варианты (по умолчанию) |

### Resume

Результаты пишутся в `benchmark/results/results.json` по модели. При повторном запуске уже
пройденные модели/фазы/варианты пропускаются (resume по файлам в `cache/<model>/` и по
`results.json`).

### Результаты

- `benchmark/results/results.json` — полные результаты по каждой модели (eval — dict по
  quantize-вариантам: `['eval']['onnx']`, `['eval']['fp16']`, `['eval']['int8']`, `['eval']['int8-calib']`)
- `benchmark/results/summary.tsv` — TSV-сводка для просмотра (метрики эталона берутся из onnx-варианта)
- `benchmark/results/benchmark.log` — лог со всеми stdout/stderr

### Точность (eval.py)

eval.py гоняется в convert-контейнере на тех же кэш-файлах, что и FPS (onnx → `--backend cpu`,
rknn → `--backend npu --rknn-suffix _fp16/_int8/_int8-calib`), и считает метрики тем же
Pipeline (letterbox, постпроцесс, NMS), что и в продакшене. Вручную — одной моделью:

```bash
cd benchmark
python eval.py detect ../datasets/coco_val2017 \
    --model-dir ../cache/yolo11n --model-name yolo11n --backend cpu
```

Сопоставление задач (models.py → eval.py): `detect`→`detect`, `seg`→`segment`,
`cls`→`classify`, `pose`→`pose`, `obb`→`obb`, `sem`→`semantic`.

## Стресс-тест

Нагрузочный тест CPU + NPU + RAM с мониторингом температур и загрузки.
Не требует root — обычный пользователь из группы `docker` может запускать напрямую.
Root необходим только для режима `--overclock`.

```bash
# TUI-режим (по умолчанию): Unicode-бары, per-core CPU, температуры с градиентом
./stress_test.sh

# Компактный табличный режим (старый формат)
./stress_test.sh --compact

# 5 минут, макс. нагрузка на NPU (16 потоков)
./stress_test.sh -d 300 -s 16

# Только NPU, 10 минут
./stress_test.sh --npu-only -d 600

# С логом в CSV
./stress_test.sh -d 3600 -l stress_results.csv
```

| Опция | По умолчанию | Описание |
|-------|-------------|----------|
| `-d, --duration SEC` | `0` (до Ctrl+C) | Длительность |
| `-c, --npu-cores` | `0,1,2` | NPU ядра |
| `-s, --npu-streams` | `8` | Потоков инференса на ядро |
| `-m, --model` | `model/model.rknn` | RKNN модель |
| `-i, --interval SEC` | `2` | Интервал мониторинга |
| `-l, --log FILE` | — | CSV лог |
| `--npu-only` | — | Только NPU |
| `--cpu-only` | — | Только CPU + RAM |
| `--compact` | — | Компактный табличный режим (без Unicode-баров) |

TUI-режим (по умолчанию):
- Кластеры CPU: policy0 (LIT A55), policy4 (BIG0 A76), policy6 (BIG1 A76) с per-core sub-барами
- NPU: aggregate load bar (debugfs, fallback devfreq)
- RAM: usage bar
- Температуры: 3-цветный градиент (зелёный → жёлтый → красный) + throttle-индикаторы

Ctrl+C корректно останавливает все процессы и выводит финальные температуры.

Также доступен Python-аналог: `python3 stress_test.py` (опции аналогичны).

## Структура

```
percept/                     # Модули приложения
├── inference/               # Загрузка модели и инференс (RKNN/ONNX)
│   └── __init__.py          # ModelBackend: загрузка, infer, detect, reload
├── perception/              # Постпроцесс
│   └── __init__.py          # dfl(), nms()
└── io/                      # Ввод/вывод изображений
    └── __init__.py          # letterbox(), scale_coords()

main.py                      # Точка входа (async)
app.py                       # Flask HTTP сервер (/frame, /objects, /reload, /test)
pipeline.py                  # Оркестрация: загрузка постпроцесса + вызов ModelBackend
config.py                    # Конфигурация (CLI/env)
camera_source.py             # Абстракция камеры (USB/GigE/RTSP/V4L2/WebRTC/GStreamer)
video_loop.py                # Async capture → detect → notify loop
websocket_server.py          # WebSocket push для live viewer
detector_frame.py            # DetectorFrame dataclass
docker-compose.yml           # Сервис webcam
Dockerfile                   # Образ (rknn-toolkit-lite2 + OpenCV + pypylon)
model/                       # Активная модель (.rknn/.onnx + _postprocess.py + model_name.txt + quantize.txt)
templates/
├── index.html               # Live viewer (WebSocket + canvas)
└── test.html                # Тестовая страница (галерея + drag-drop)
test/                        # Тестовые изображения
start.sh                     # Запуск сервера
stop.sh                      # Остановка сервера
deploy.sh                    # Деплой модели (конвертация + запуск, --int8 для калибровки)
setup_datasets.py            # Загрузка датасетов (task --mini/--full, или --coco/--dota/--cityscapes/--imagenet)
oc_manager.sh                # Менеджер разгона (test/apply/revert/monitor/set)
oc_test.sh                   # Тест разгона (пошаговый переход по частотам)
stress_test.sh               # Стресс-тест NPU/CPU/RAM (bash, Ctrl+C)
stress_test.py               # Стресс-тест NPU/CPU/RAM (python)
bench_overclock.sh           # Бенчмарк с разгоном CPU/NPU/GPU (FPS до → разгон → FPS после)
overclock/
└── rk3588-opi5pro-oc.dts    # Device tree overlay для разгона Orange Pi 5 Pro
benchmark/
├── benchmark.py             # Оркестратор: --phase convert|validate|bench|eval|all
├── convert_bench.py         # Фаза 1: экспорт + компиляция (4 RKNN/ONNX + postprocess на модель)
├── validate_bench.py        # Фаза 2: ONNX vs RKNN FP16+INT8 валидация (NPU/симулятор)
├── fps_bench.py             # Фаза 3: FPS + per-core CPU/NPU нагрузка
├── eval.py                  # Фаза 4: точность (все quantize-варианты, все 6 задач)
├── models.py                # Списки моделей (FULL: 111, PILOT: 21)
├── run_3way_eval.py         # Interleaved 3-way проверка: ultra-NMS vs std vs RK-opt постпроцесс
├── bench.sh                 # Обёртка для benchmark.py
└── results/                 # Результаты (results.json, summary.tsv, benchmark.log)
convert/
├── convert.py               # Конвертер YOLO → ONNX → RKNN (с кешем)
├── templates/               # Шаблоны постпроцесса (7 задач: det/seg/cls/pose/obb/sem/detr)
├── patches/                 # RKNN-патчи
├── weights/                 # Исходные .pt
├── docker-compose.yml       # Docker compose конвертера
├── Dockerfile               # Образ конвертера (надстройка над percept-percept:latest)
└── model/                   # Активная модель для сервиса (.rknn/.onnx + postprocess)
cache/                       # Кеш конвертера: cache/<model>/{model}.pt|onnx|_fp16.rknn|_int8.rknn|_int8-calib.rknn|_postprocess.py
datasets/                    # Датасеты (скачиваются on-demand)
├── coco_val2017/            # COCO val2017 — detect, seg, pose (~19 GB)
├── dotav1/                  # DOTAv1 — obb (~1.9 GB)
├── cityscapes/              # Cityscapes — semantic
└── imagenet/                # ImageNet val — classify (~6.3 GB)
```

## Конфигурация

| Переменная | По умолчанию | Описание |
|------------|-------------|----------|
| `BACKEND` | `npu` | `npu` или `cpu` |
| `MODEL` | (из `model_name.txt`) | Имя модели (без расширения). Если не задано, читается из `model/model_name.txt` |
| `PORT` | `5000` | HTTP порт |
| `CAM_MODE` | `auto` | Режим камеры |
| `CAM_SOURCE` | — | Источник (IP/USB/path) |
| `CAM_W` / `CAM_H` | `640` / `480` | Разрешение камеры |
| `JPEG_QUALITY` | `85` | Качество JPEG (1–100) |
| `LOG_LEVEL` | `INFO` | Уровень логирования |
| `MASK_FORMAT` | `contour` | Формат масок (contour/polygon) |
| `TEST_MODE` | `1` | Тестовый режим (файловый ввод) |

## Доступные модели

### Ultralytics YOLO (111 моделей)

| Семейство | Размеры | Задачи | Количество |
|-----------|---------|--------|------------|
| YOLOv3 | u, sppu, tinyu | detect | 3 |
| YOLOv5 | nu/su/mu/lu/xu, n6u/s6u/m6u/l6u/x6u | detect | 10 |
| YOLOv8 | n/s/m/l/x | detect, seg, cls, pose, obb | 25 |
| YOLOv9 | t/s/m/c/e | detect | 5 |
| YOLOv9 | c, e | seg | 2 |
| YOLOv10 | n/s/m/b/l/x | detect | 6 |
| YOLO11 | n/s/m/l/x | detect, seg, cls, pose, obb | 25 |
| YOLO12 | n/s/m/l/x | detect | 5 |
| YOLO26 | n/s/m/l/x | detect, seg, cls, pose, obb, sem | 30 |

## Эндпоинты

| Route | Метод | Описание |
|-------|-------|----------|
| `/` | GET | Live viewer (HTTP poll, SVR/CLT FPS) |
| `/frame` | GET | Последний обработанный JPEG |
| `/objects` | GET | JSON: детекции + fps + frame_id + model |
| `/reload` | GET | Перезагрузка модели (`?model=name&backend=cpu`) |
| `/test` | GET | Тестовая страница (галерея + drag-drop) |
| `/test/images` | GET | JSON: список тестовых изображений |
| `/test/images/<name>` | GET | Отдача тестового изображения |
| `/test/images/<name>` | DELETE | Удаление из галереи |
| `/test/upload` | POST | Загрузка изображения в галерею |
| `/process` | POST | Обработка загруженного файла |

## Мониторинг NPU

```bash
# Нагрузка + частота (всегда доступно)
cat /sys/class/devfreq/fdab0000.npu/load
# Вывод: 100@1000000000Hz  (нагрузка% @ частота)

# Только частота
cat /sys/devices/platform/fdab0000.npu/devfreq/fdab0000.npu/cur_freq

# Нагрузка по ядрам (требует debugfs, может быть пусто)
cat /sys/kernel/debug/rknpu/load

# Непрерывный мониторинг
watch -n1 'cat /sys/class/devfreq/fdab0000.npu/load'
```

## Известные ограничения

- YOLO-NAS, YOLOWorld не поддерживаются
- Компиляция RKNN на aarch64 медленная для больших моделей (>100 MB)
- RT-DETR runtime init падает на NPU (используется CPU)
- `quantize=16` (default) = float-модель; INT8 только с `quantize=8` + calibration
- INT8 калибровка на 512 изображениях
- GigE камера: первый захват может быть медленным (GVCP discovery)
- USB камера на WiFi: HTTP poll может давать ~5fps из-за RTT
- **RK3588S shared bus**: NPU инференс DMA может вызывать микро-глитчи USB камеры (~50мс). Код.camera_source.py` автоматически повторяет чтение при первом сбое перед закрытием камеры
- **Питание**: слабый БП вызывает brownout при одновременной нагрузке CPU+NPU+камера → перезагрузка. Рекомендуется БП ≥5V/4A
- **ONNX CPU на RK3588S**: инференс через CPU потребляет слишком много энергии и вызывает brownout. Используйте всегда `BACKEND=npu`
- **Конвертер Docker**: конвертация и eval в convert-контейнере идут на реальном NPU (`/dev/dri`), поэтому не запускайте их одновременно с работающим сервисом percept (конфликт NPU). Артефакты лежат в `cache/<model>/`, переносимы между сборками
- **GPU stress**: Mali-G610 использует проприетарный Bifrost kernel driver (DDK g21p0) без userspace библиотеки. Vulkan ICD (Panthor/Panfrost) не работает с проприетарным ядром. EGL Mesa fallback-ит на llvmpipe (CPU). GPU stress недоступен без `libmali.so` от Rockchip той же DDK-версии. CPU+NPU стресс покрывают ~95% peak power по общим power rail'ам

---

## Источники

Патч Detect.head (выход 9 тензоров `[box, cls, score_sum] × 3 strides`, DFL/decode/NMS на CPU) взят из форков airockchip:

- **airockchip/ultralytics_yolov8** — форк ultralytics с RKNN-экспортом для всего семейства YOLO (v5/v8/v9/v10/v11/v12). Основной источник патча Detect.head и конфигурации конвертера (mean=[0,0,0], std=[255,255,255]). См. [RKOPT_README.md](https://github.com/airockchip/ultralytics_yolov8/blob/main/RKOPT_README.md)
- **airockchip/rknn_model_zoo** — референсные постпроцессы (Python + C++) и eval-пайплайн для RKNN-моделей (pycocotools mAP). Источник `score_sum` ранней фильтрации и архитектуры C++ постобработки
- **airockchip/rknn-toolkit2** — конвертер ONNX→RKNN и RKNNLite runtime. Уже используется в проекте для установки `librknnrt.so` и `rknn_server` (см. раздел «Предустановка»)

## Что сделано (июль 2026)

### Полная русификация проекта
Все пользовательские сообщения, логи, docstrings, RuntimeError переведены на русский:
- `convert/convert.py` — конвертер, INT8 калибровка, dataset.txt логирование
- `percept/inference/__init__.py` — пайплайн NPU/CPU
- `pipeline.py`, `main.py`, `video_loop.py`, `camera_source.py`
- `websocket_server.py`, `detector_frame.py`
- `convert/patches/rknn_patch.py` — патч Detect.head для RKNN (по [airockchip/ultralytics_yolov8](https://github.com/airockchip/ultralytics_yolov8))
- `benchmark/convert_bench.py`, `benchmark/fps_bench.py`

**Не переводили:** HTTP ответы в `app.py`, `argparse` help, логи библиотек (ultralytics, RKNN toolkit, waitress).

### Docker
- Образ конвертера: `percept-converter:latest` — **надстройка над `percept-percept:latest`**
  (наследует python 3.11, rknn-toolkit-lite2, cv2/numpy/ffmpeg), сверху только тестовые
  зависимости: ultralytics, rknn-toolkit2, onnxruntime, onnxslim, pycocotools
- Конвертер работает на aarch64 Orange Pi 5 Pro, RKNN-инференс на **реальном NPU** через rknnlite
  (rknn-toolkit2-эмулятор — только fallback для x86)
- Образ прод: `percept-percept:latest` (сервис `percept` в docker-compose.yml)
- `docker-compose.yml`: сервис `percept`, переменные `MODEL`, `WEB_MODE`
- `convert/docker-compose.yml`: `image: percept-converter:latest`, сервис `convert`,
  проброс `/dev/dri` + хостовых `librknnrt.so`/`librknn_api.so`
- `config.py` читает `model/model_name.txt` если `MODEL` не задан

### Калибровка INT8
- Используются **все 5000 изображений** COCO val2017 (`datasets/coco_val2017/images/val2017/`)
- `_make_calib_yaml(task)` генерирует YAML на лету с полным `nc: 80` и `names: [...]` для всех задач (seg/semantic/obb/pose/classify)
- Исправлен баг: для segment/semantic теперь добавляются `nc` и `names` (раньше было `key missing ❌`)
- Ультралитикс `get_int8_calibration_dataloader()` читает **все** изображения из `data[split]`, без лимита, `fraction=1.0`, `batch=16`
- Прогресс-бар RKNN toolkit2 (`270/270`) — это внутренние шаги квантования, **не** количество изображений
- Добавлено логирование `dataset.txt` в `convert.py` (`_log_dataset_txt()`): ищет в `/tmp/dataset.txt`, `dataset.txt`, `calib_dataset.txt`, `*.txt` в рабочей директории — для проверки списка файлов, используемых при калибровке

### Бенчмарк
- Переписан `benchmark/benchmark.py` с флагом `--model` для запуска одной модели
- Resume-логика: `results.json` апсертится по модели; при рестарте завершённые фазы/варианты пропускаются
- Лог-файл в режиме `'a'` (сохраняется история)
- Статусы: ПРОЙДЕНО / ПРОВАЛ
- Сервис в docker-compose: `percept` (не `webcam`)
- `benchmark/bench.sh` передаёт `"$@"` в benchmark.py, останавливает percept перед запуском
- TSV-суммаризация в `benchmark/results/summary.tsv`
- 4 варианта квантизации: onnx/none(fp16)/int8/int8-calib
- Фаза eval: точность через eval.py (top1/top5 для cls, mAP для det/seg/pose/obb, mIoU для sem), все 4 варианта, в convert-контейнере
- Per-core CPU/NPU нагрузка в fps_bench.py

### deploy.sh
Переписан в однокомандный деплой:
```bash
./deploy.sh yolo26n-pose       # FP16 RKNN (быстрый старт)
./deploy.sh yolo26n-pose --fp16       # то же
./deploy.sh yolo26n-pose --onnx       # ONNX на CPU (без NPU)
./deploy.sh yolo26n-pose --int8       # конвертация INT8 + запуск
./deploy.sh yolo26n-cls --int8-calib  # INT8 с калибровкой (ImageNet)
./deploy.sh                    # только запуск (использует .rknn из /model)
./deploy.sh yolo26n-pose --full-test  # конвертация + FPS тест
```
Автозагрузка нужного датасета при первом запуске (COCO/DOTA/Cityscapes/ImageNet).
При конвертации записывает имя модели в `model/model_name.txt` и тип квантизации в `model/quantize.txt`.
Сервис читает оба файла при старте — имя модели и тип отображаются в заголовке вкладки браузера и в ответе `/objects`.
Переменные окружения: `MODEL`, `WEB_MODE`.

### Стресс-тест
- `stress_test.sh` — bash-скрипт: NPU (через Python rknnlite) + CPU/RAM (stress), мониторинг температур, NPU load, CPU load
- `stress_test.py` — python-аналог с теми же возможностями
- Ctrl+C корректно останавливает все процессы
- Легенда столбцов выводится в шапке таблицы
- CPU%: расчёт через дельту `/proc/stat` (файл-состояние для обхода subshell)
- NPU: до 100% загрузки (3 ядра × 8 потоков, yolo26x-seg 66 МБ)

### Разгон (`bench_overclock.sh`, `oc_manager.sh`, `oc_test.sh`)

Менеджер разгона с проверкой стабильности по FPS. Автоматически определяет нестабильность.

#### Бенчмарк с разгоном:

```bash
sudo ./bench_overclock.sh                              # текущая модель
sudo ./bench_overclock.sh --model yolo11x-pose         # конкретная модель
sudo ./bench_overclock.sh --measure 30                 # дольше измерение (по умолчанию 20с)
sudo ./bench_overclock.sh --quantize int8              # только INT8
```

> При первом запуске скрипт автоматически устанавливает OC DTB overlay
> (разблокировка 2352/2400 МГц) и перезагружает систему.
> После ребута повторите ту же команду — начнётся бенчмарк.

#### Откат к стоковым частотам:

```bash
sudo ./bench_overclock.sh --revert
```

#### Два типа FPS-замеров:

- **Camera FPS** — реальная работа сервиса с камеры ( замер до и после разгона)
- **Inference FPS** — воспроизводимый замер с файлами с диска ( замеры между частотами, `pause/unpause` сервиса)

Сервис НЕ останавливается — на время inference-замеров делается `docker pause/unpause`.

#### Управление разгоном:

```bash
./oc_manager.sh status              # текущее состояние (без sudo)
sudo ./oc_manager.sh test           # тест всех частот + поиск макс. стабильной
sudo ./oc_manager.sh test --quick   # быстрый тест (стоп на первом FAIL)
sudo ./oc_manager.sh apply          # применить сохранённую конфигурацию
sudo ./oc_manager.sh revert         # откат к стоковым частотам
sudo ./oc_manager.sh set cpu=2400 npu=1000 gpu=800  # установить конкретные частоты
sudo ./oc_manager.sh monitor        # мониторинг + автооткат при нестабильности
sudo ./oc_test.sh --compare         # FPS на всех частотах (до и после DTB OC)
```

**Как работает проверка стабильности:**
- Замеряет FPS на стоковых частотах
- Устанавливает каждую частоту, замеряет FPS
- Если FPS падает >5% от baseline — считается нестабильным
- Разгон оставляется включённым после бенчмарка (откат через `--revert`)

**Откат DTB:**
```bash
sudo cp /boot/dtb/rockchip/rk3588s-orangepi-5-pro.dtb.bak \
         /boot/dtb/rockchip/rk3588s-orangepi-5-pro.dtb
sudo reboot
```

### Удалённые файлы
- `run.sh` (legacy) — удалён
- `model/yolo11n_postprocess.py` — root-owned, требует sudo для правки
