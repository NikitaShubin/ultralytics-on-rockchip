Все общение с пользователем должно вестись только на русском языке.

## Правила проекта

### Данные — не в git
Все тестовые данные (датасеты, артефакты бенчмарков, логи) скачиваются скриптами on-demand.
Запрещено коммитить: `benchmark/data/`, `benchmark/results/`, `convert/datasets/`, `cache/`, логи, `.tsv`-результаты.
Скрипты автозагрузки: `setup_datasets.py`, `benchmark/setup_datasets.py`.

### Относительные пути
Все пути в скриптах и конфигах — относительно корня проекта (`$(dirname "$0")` или `./`).
Исключение: системные хостовые пути (`/boot/dtb/...`, `/dev/...`, `/usr/lib/...`).

### Долгие процессы — в фон
Любые процессы, занимающие больше ~30 секунд (eval, бенчмарки, конвертация моделей, скачивание),
запускать в фон (`& disown`), а не блокируя терминал.
Логировать вывод в файл (`> /tmp/xxx.log 2>&1 & disown`).
Периодически проверять аутпут через `tail` или `grep`.

## Work State (автообновляемое)

### Objective
**Эквивалентность eval.py vs ultralytics .val (col1/col2/col3)**. Проверено для yolo11n (все 5 задач),
yolo26n (все 6 задач), yolo12n (detect) и всей линейки real-time моделей — все три колонки совпадают.
RKNN-деплой отложен по решению пользователя. Пайплайн и постпроцесс — часть проекта; постпроцесс обязан
использоваться в конверторе (шаблоны convert/templates/). Полный план и статус — в `PLAN.md`
(источник истины для следующего агента).

### Инфраструктура
- **convert-образ = НАДСТРОЙКА над percept-percept:latest (2026-08-09)**: `convert/Dockerfile` теперь
  `FROM percept-percept:latest` (а не `python:3.11-slim`) → наследует rknn-toolkit-lite2, cv2/numpy, ffmpeg,
  тот же python 3.11; сверху добавляются ТОЛЬКО тестовые зависимости (ultralytics, rknn-toolkit2, onnxruntime,
  onnxslim, pycocotools). Ноль дублей с percept: обновления percept (rknnlite и т.п.) попадают в convert
  автоматически при пересборке. **convert-контейнер инферит rknn на РЕАЛЬНОМ NPU**
  (устройство — Orange Pi 5 Pro, RK3588, aarch64, НЕ x86!): в `convert/docker-compose.yml` добавлены
  volumes хостовых `librknnrt.so`/`librknn_api.so` в `/usr/lib/` (как в percept) + проброшен `/dev/dri`.
  `ModelBackend` (percept/inference/__init__.py) берёт `rknnlite` первым → реальный NPU, `rknn-toolkit2`
  (эмулятор) — только fallback, если rknnlite недоступен. **ПОЛЬЗОВАТЕЛЬ ОТКЛОНИЛ дампы preds через npz**
  (долго + износ носителя) — метрики считаются там же, где инференс.
- **Benchmark eval-фаза**: `benchmark/benchmark.py` гоняет `benchmark/eval.py` в convert-контейнере
  (`docker compose run`): `onnx → --backend cpu`, `fp16/int8/int8-calib → --backend npu --rknn-suffix _fp16/_int8/_int8-calib`.
  `results.json['eval']` — dict по вариантам (onnx/fp16/int8/int8-calib); TSV берёт эталон из onnx-варианта.
  `eval_bench.py` УДАЛЁН.
- **eval.py CLI**: добавлен `--rknn-suffix` (прокинут в Pipeline → ModelBackend).
- **Сопоставление задач**: models.py (`detect/seg/cls/pose/obb/sem`) → eval.py (`detect/segment/classify/pose/obb/semantic`)
  через `TASK_TO_EVAL`; eval.py ожидает КОРЕНЬ датасета (не images/val2017) → новый `_container_dataset_root()`.
- **`cache/yolo11n/yolo11n.onnx` СЛОМАН (перезаписан std-версией 2026-08-09, 1 выход вместо RK-opt 9)**:
  RK-opt постпроцесс `yolo11n_postprocess.py` (per-stride) с ним падает (IndexError). При необходимости
  восстановить: `ensure_rko_cache('yolo11n', ..., force=True)` (перегенерит RK-opt ONNX через convert).
- **Мини-датасеты**: `datasets/{coco128,coco8-pose,imagenet100,...}` добавлены в `.gitignore`.

### Interleaved runner (benchmark/run_3way_eval.py)
- Один цикл по кадрам: на каждом кадре col1 (ultralytics nms на том же std-тензоре), col2 (наш std
  постпроцесс), col3 (наш RK-opt постпроцесс); сразу сравнивать preds, копить только tp-агрегаты и счётчики
  расхождений; при расхождении сохранять только эти изображения (diverged-список).
- Конфликт Pipeline решён: `Pipeline._load_postprocess` грузит глобальный `sys.modules['postprocess_mod']` —
  поэтому работаем с `ModelBackend` напрямую и загружаем постпроцессы под уникальными именами (`pp_std`, `pp_rko`).
- col1 = `non_max_suppression(max_time_img=1000)` из `ultralytics.utils.nms` (НЕ ops) на std_raw
  (letterboxed-пространство); col2/col3 — наш постпроцесс (orig) → перевод в letterboxed через `x*scale+pad_left`.
- Сравнение с допуском: score 1e-6, bbox 1e-3, angle 1e-4, kpts 1e-3, mask 0.01; class_id строго.
- Метрики — инкрементальные per-image tp (`_box/_pose/_seg/_obb_tp_per_image` + `_ap_finalize(prefix=...)`);
  pycoco в интерливе не нужен.
- classify/semantic: col1 == col2 по построению (тот же std_raw); semantic — инкрементальная confusion-матрица
  per-image (OOM-фикс, см. ниже).
- Результат: `cache/3way_results/<model>-<task>_interleaved.json` (col1/col2/col3, diverged-счётчики, equal-флаги, preds).
- **ВАЖНО: имена моделей для задач** — std_model_dir/rko_model_dir строятся из МОДЕЛИ, а не task:
  `yolov8n-seg segment`, `yolov8n-pose pose`, `yolov8n-cls classify`, `yolo11n-obb obb`, `yolo26n-sem semantic`
  (иначе загружается detect-кэш `yolov8n_std` → `std_raw[1]` IndexError).
- **НАЙДЕН КРИТИЧЕСКИЙ БАГ nms-мутации**: `non_max_suppression` делает `prediction[..., :4] = xywh2xyxy(...)`
  через `transpose`-вью инплейс; `torch.from_numpy(np.asarray(std_raw[0]))` БЕЗ копии → nms молча портит буфер
  std_raw (xywh→xyxy). `_nms_det` обязан использовать `.copy()` (run_3way_eval.py:678), иначе последующий
  `std_pp.postprocess(std_raw)` применяет xywh2xyxy повторно → сломанные bbox.
- **Diverged-кадры при near-tie**: у ultralytics-NMS и нашего NMS при почти/равном score (разница ≤ 1e-6)
  редкие кадры выбирают разных кандидатов; повторы могут давать разное число preds (недетерминизм torch-NMS).
  Метрики при этом НЕ меняются.

### Инфраструктура docker
- **docker работает без sudo** (группа docker). Ad-hoc команды (eval, smoke-тесты)
  гонять через convert-сервис (тот же `percept-converter:latest`, NPU-доступ):
  `docker compose -f convert/docker-compose.yml run --rm --entrypoint python convert <script> ...`
  (у образа ENTRYPOINT=convert.py — для произвольных команд нужен `--entrypoint`).
  Фоновые прогоны: `nohup ... > /tmp/xxx.log 2>&1` или `docker exec -d` в одноразовом
  контейнере, запущенном `docker compose run`; лог — в bind-папке cache/.
  (bash-инструмент таймаутит 120s и убивает фоновые процессы — запускать через tmux).
  Контейнер `percept` (percept-percept) НЕ подходит (нет ultralytics).
  **segsmoke УДАЛЁН (2026-08-11)** — был дублем convert-сервиса.

### Important Details
- **System freezes under sustained CPU load** (even 2 containers) due to 8GB RAM exhaustion + thermal throttling. Workaround: run 1 container at a time with `--memory=4g` or for eval use `--limit 500`.
- **YOLO26 export — end2end по умолчанию**: ультралайтс 8.4.105 экспортирует yolo26 в ONNX с NMS в графе (выход `[1,300,6]`), НЕ raw. `.val()` умеет оба формата: `DetectionPredictor.postprocess` → `non_max_suppression(..., end2end=model.end2end)`. Для честного сравнения (как yolo11) std ONNX экспортируется с `end2end=False` → raw (`benchmark/export_std_onnx.py:58`).
- **YOLO26 — НЕТ DFL**: box = 4 канала (ltrb-отступы) напрямую; `reg_max=1` → `NO_DFL=True` в шаблонах. RK-opt per-stride: тройка [box(4), cls(C, sigmoid), sm(1)] × nl; seg добавляет coeff(32)+proto(160); sem = один выход (см. ниже).
- **yolo26 RK-opt выходы**: detect [1,4]+[1,C]+[1,1] per stride; pose [1,4]+[1,1]+[1,51] (nc=1 person); seg [1,4]+[1,C]+[1,32] ×3 + proto[1,32,160,160]; obb [1,4]+[1,C]+[1,1] (angle в box, GRID [128,64,32], 1024); cls [1,1000] на 224; sem [1,19,1024,1024] (logits) для RK-opt, std sem = argmax-карта [1,1024,1024] (bake_argmax в SemanticSegment). semantic_postprocess обрабатывает ОБА формата (ndim==3 → argmax по каналу; ndim==2 → карта как есть).
- **RK-opt патчи = полная сеть** (airockchip-стиль): головы ОСТАЮТСЯ в графе (NPU), на CPU выносится только декодирование (DFL/dist2bbox/grid) + NMS. Это НЕ «отделение головы от бэкбона».
- **Primary goal: eval.py on std ONNX must give EXACT same metrics as ultralytics .val() on the same model.** RK-opt is secondary priority.
- **RK-opt ONNX models have sigmoid baked into** `cls_scores` outputs (values are in 0-1 range). Postprocess must NOT apply `sigmoid()` to cls_scores. Only kpts confidence needs sigmoid.
- **RK-opt pose keypoint decoding must use YOLO11 standard formula**: `(offset * 2.0 + grid - 0.5) * stride`. The `if True:` branch (simple offset) is wrong.
- **Classify preprocessing**: ultralytics uses PIL Resize(224, bilinear) + CenterCrop(224) + ToTensor(/255). Pipeline's default letterbox is wrong for cls.
- **`/255.0` bug fixed**: `ModelBackend.infer()` no longer applies `/255.0` unconditionally. Instead, `/255.0` is applied only in the default letterbox preprocessing path. `custom_preprocess` returns [0,1] tensor directly.
- **val-NMS ultralytics — multi_label=True**: `models/yolo/detect/val.py postprocess` вызывает `nms.non_max_suppression(..., multi_label=True, nc=self.nc, rotated=True)`. Это даёт per-box-per-class расширение (каждый бокс × каждый класс с conf>thresh), т.е. у val БОЛЬШЕ предсказаний, чем у single-label постпроцесса. eval-постпроцессы detect/obb обязаны использовать multi-label expand (`np.where(cls > CONF_THRESH)` → x=[box[i], cls[i,j], j]) + class-aware NMS (offset `cls_id*MAX_WH`).
- **ЗАПРЕЩЕНО форсировать imgsz при экспорте/валидации** (2026-08-06): рабочее разрешение берётся из исходного .pt (args.imgsz) — конвертер так и делает (export без override), Pipeline детектит вход из модельного файла. Еval НЕ заменяет разрешение: eval.py строит тот же `Pipeline`, input_size из ONNX shape (percept/inference/__init__.py:88-89) или INPUT_SIZE постпроцесса (:93-113).
- **Аудит входов ONNX (2026-08-06)**: все cache-модели на нативном разрешении КРОМЕ obb (640 vs 1024). Мусорные не-нативные ONNX верхнего уровня удалены.
- **ВНИМАНИЕ (2026-08-07): `/tmp/opencode` ОЧИЩЕН при перезагрузке** — все бэкапы и ad-hoc скрипты утеряны. Не полагаться на /tmp/opencode.

### Completed (технические фиксы, без значений метрик)
- **ПРОБЛЕМА one2one РЕШЕНА (2026-08-07)**: yolo26 end2end-модели: ультралайтс `model.fuse()` (Exporter, exporter.py:801-817) для end2end-голов УДАЛЯЕТ cv2/cv3 (one2many) и оставляет ТОЛЬКО one2one_cv2/one2one_cv3 → наш патч `_resolve_heads` неявно брал one2one → RK-opt давал другие логиты. **Фикс**: в `export_onnx_only`/`export_and_compile` перед export ставим `m.end2end=False` на всех модулях + `model.end2end=False` → fuse() оставляет cv2/cv3, RK-opt = one2many как std. Переэкспортированы все 6 RK-opt ONNX (проверено: model.23 cv3=30, one2one=0).
- **YOLO12n = стандартная голова Detect** (reg_max=16, DFL, end2end=False, выход [1,84,8400]) — RK-opt патчи и шаблоны yolo11 подошли БЕЗ изменений; отличие только в backbone (A2C2f attention). Официальных весов yolo12 для seg/pose/obb/cls НЕТ — только detect n/s/m/l/x.
- **Pose: GT в letterboxed-пространстве**: у ultralytics `Format(normalize=True)` нормализует bboxes/kpts по letterboxed-изображению 640×640 (LetterBox уже применил scale+pad); `_prepare_batch` умножает на imgsz → GT в letterboxed 640-пространстве. GT маппится как `orig_px*scale + pad` (НЕ `orig_px/iw*640`). letterbox percept `round`→`ceil` (ultralytics использует `min(math.ceil(w*scale), imgsz)`) → входы pixel-идентичны.
- **Segment: mask_overlap=True по умолчанию**: SegValidator val использует `Format(mask_overlap=True)` → `polygons2masks_overlap((640,640), letterboxed_segments, ratio=4)` даёт один (160,160) int-слой со значениями 1..N (1 = инстанс с максимальной площадью, N = минимальной; маленькие рисуются поверх больших через `np.maximum`). Инстансы и cls сортируются по площади (desc). `_prepare_batch` декодирует per-instance: `masks = (overlap == i+1)`. Метрика: `mask_iou` + `match_predictions` + `ap_per_class`. Реализовано в `compute_inmemory_seg_map`.
- **OBB**: GT `rboxes = xyxyxyxy2xywhr` (orig) → letterboxed `bbox[..., :4].mul_(imgsz[[1,0,1,0]])` (cx,cy,w,h×640); метрика `batch_probiou` + `match_predictions` + `ap_per_class`. `ratio_pad` в pbatch = `((gain,gain),(pad_x,pad_y))`. Реализовано в `compute_inmemory_obb_map` + `xyxyxyxy2xywhr_np` (numpy-реплика cv2.minAreaRect, правило w<h swap).
- **obb col1**: ультралайтс NMS имеет time_limit (2.0+max_time_img*bs) и под CPU-нагрузкой урезает preds. Для честного col1 задавать max_time_img=1000 (monkeypatch nms.non_max_suppression) или сравнивать по capture preds. Ультралайтс не определяет task для obb-ONNX — нужно `YOLO(onnx, task='obb')`, иначе val идёт как detect (нули).
- **sem-этап**: eval.py semantic ветка реализована (`compute_inmemory_sem_map` — реплика SemanticMetrics: confusion [gt,pred], pad GT=255, mIoU только по present классам, pixacc) + `letterbox_semantic_gt` (NEAREST к new_unpad=isz-pad + copyMakeBorder 255) + `load_cityscapes` + `apply_cityscapes_label_mapping`. Порог определения raw mask: `gt.max() > 18` (trainId всегда 0..18). **ВАЖНО для col3 (ИСПРАВЛЕНО 2026-08-07)**: `_rknn_semantic_segment` (convert/patches/rknn_patch.py:96) РАНЬШЕ заменял bilinear-апсэмпл на nearest → col3 != col2. **Фикс: переведён на bilinear align_corners=False (как std)**.
- **OOM-фикс semantic (исправлен)**: semantic-ветка копила все pred/gt-карты (1024² int64) → OOM-kill; фикс — инкрементальный confusion matrix per-image + `sem_metrics_from_matrix()`.
- **Метод col1 sem подтверждён**: `SemanticSegmentationValidator` ЕСТЬ в 8.4.105 (models/yolo/semantic/val.py), `.val()` на std ONNX (task='semantic') работает и даёт mIoU байт-в-байт == eval.py. Ключи результата: 'metrics/mIoU' и 'metrics/pixel_acc'.
- **EVAL_PRECISE=True** (detect/seg/obb): bbox float, БЕЗ `int()`/`clip()` — метрика ultralytics не клипает. eval-режим: `MIN_WH=1, MAX_DET=300, DEDUP_ENABLED=False`.
- **OBB эквивалентность**: фикс multi_label=True NMS в obb-постпроцессе (single-label давал меньше preds и заниженную метрику). GT-маппинг per-axis (sx=(isz-pad_w)/w, sy=(isz-pad_h)/h).
- **Поза equivalence**: исправлены БИТЫЕ yolo-лейблы (регенерация) — прежняя дельта была из них.
- **DETECT эквивалентность**: `EVAL_PRECISE=True` в std-постпроцессе (float bbox без int()/clip — ultralytics метрика не клипает).
- **SEG эквивалентность**: `r.box.map50` — это BOX-метрика, НЕ mask! mask-метрика из tp_m. Pred-маски байтово идентичны.
- **cls col2**: two bugs fixed — (1) wrong preprocessing (PIL Resize(224)+CenterCrop(224)+ToTensor vs letterbox), (2) double `/255.0` in `infer()`.
- **detect eval mode**: now sets `MIN_WH=1, MAX_DET=300, DEDUP_ENABLED=False` to match ultralytics postprocessing. Previously `MIN_WH=40` filtered small objects.
- **`load_yolo_as_coco` keypoints**: now extracts 17 keypoints (x,y,v) from YOLO pose labels (56-value format: `class cx cy w h x1 y1 v1 ... x17 y17 v17`). Previously ignored keypoints entirely — caused pose eval to return all -1.0.
- **col1 падал с "Dataset ... images not found, missing path .../convert/datasets/coco8/images/val"**: `datasets/coco8` был симлинком на абсолютный хостовый путь → внутри контейнера битый. **Фикс**: заменить симлинк на реальную копию. УРОК: в `datasets/` не должно быть симлинков на абсолютные хостовые пути — только реальные директории.
- **Parallel NMS в eval.py (2026-08-10)**: полный eval fp16/onnx на 5000 кадров упирался в
  torchvision NMS на 30000 кандидатов: при conf=0.001 fp16-шум > 0.001 → expand даёт 672000 кандидатов
  → NMS = **1.35с/кадр** на ARM (у ultralytics `.val()` ТО ЖЕ САМОЕ — тот же `torchvision.ops.nms`
  на тех же 30000; int8 быстрый, т.к. квантование обнуляет шум). **Фикс**: детект-ветка eval.py
  переведена на конвейер — infer (NPU) последовательно в главном потоке, pre-NMS шаги
  (decode/expand/sort) там же, NMS+reverse+format — в `ThreadPoolExecutor` по числу "больших" ядер
  (`_big_core_count()` — авто через `/sys/devices/system/cpu/cpu*/cpufreq/cpuinfo_max_freq`, на RK3588 = 4×A76;
  8 воркеров ХУЖЕ из-за 4×A55: замер 4=11.8с vs 8=17.2с на 8 кадрах). torchvision NMS освобождает GIL →
  потоки дают реальный параллелизм. Ускорение: 1500→507мс/кадр (2.7x, 93% от теоретического предела
  для 4 ядер), полный fp16 eval yolo26n = **42 мин** вместо ~2ч. Результаты байт-в-байт совпадают с
  шаблонным `postprocess` (сверка в prof_one_frame.py: OK). `_detect_pre_nms`/`_detect_nms_finish` —
  точная копия шагов detect_postprocess.py:137-203. Батч-инференс не встроен (см. ниже).
- **Эксперимент батч-инференса (2026-08-10)**: `benchmark/prof_batch_infer.py` собрал std ONNX batch=8
  → RKNN fp16 static [8,3,640,640] (в /tmp/bs8_test_*, рабочие файлы НЕ тронуты) и замерил на реальном
  NPU: bs8×1 = 470ms на 8 кадров vs bs1×8 = 649ms → **1.38x** на yolo26n (nano — NPU недогружен и на
  batch-1; для x-линейки, где NPU насыщен, прирост больше). Для eval батч дал бы лишь ~1.4x сверху
  (NMS-пул всё равно доминирует ~340мс/кадр); в продакшене (conf=0.25) полезнее: infer 81→59ms/кадр.
  Конвертер под батч НЕ переделывали.

### Interleaved runner — исправленные баги (2026-08-08)
(1) **nms-мутация std_raw**: `_nms_det` без `.copy()` портил буфер std_raw (xywh→xyxy инплейс) → col2 постпроцесс ломался. Фикс `.copy()` в run_3way_eval.py:678. (2) **'masks' vs 'masks_160'**: canon_col1 писал 'masks', а canon_objs/_seg_tp_per_image/compare_preds ждали 'masks_160' → KeyError. Унифицировано на 'masks_160'. (3) **segment/pose постпроцессы клипали bbox ВСЕГДА** (не под `if not EVAL_PRECISE` как detect) → col1 vs col2 bbox diff. Фикс в шаблонах `convert/templates/{segment,pose}_postprocess.py` + патч сгенерированных кэш-файлов. (4) **semantic-ветка** использовала постпроцесс (возвращает orig-пространство) при GT в letterboxed → TypeError 'dict'; переведена на raw argmax (`raw[0][0].argmax(0)`) как eval.py (run_3way_eval.py:633-647).

### Решение пользователя (2026-08-10): старый пайплайн остаётся основным
- **НЕ переходим на новый ultralytics-RKNN пайплайн (B-путь) полностью**: точность int8 в нём падает
  (B fp32=0.542→int8=0.435, −20%) против нашего RK-opt (A fp32=0.523→int8=0.518, −1%).
  Старый пайплайн (`percept/` + RK-opt постпроцесс) — по умолчанию.
- **Производительность НЕ страдает** (замер на yolo11n, 640, NPU, conf=0.25):
  наш infer=43.3ms + постпроцесс=14.7ms (25%) = 57.9ms (~17 fps);
  их infer=47.0ms + NMS=9.0ms = 56.0ms (~18 fps). Разница ~2ms (3%) — узкое место NPU,
  не CPU-постобработка. Их NMS тоже на CPU (декодирование у них в графе).
  Eval-режим (conf=0.001, multi-label) НЕ репрезентативен: постпроцесс раздувается до секунд
  из-за 8400×80 кандидатов → замерять скорость только в продакшен-конфиге (conf=0.25).
- **Восстановимость пайплайна при обновлении ultralytics — высокая**: рантайм `percept/`
  НЕ импортирует ultralytics (только rknnlite+numpy) → продакшен не ломается.
  Зависимость только в экспорте/бенчмарках: `rknn_patch.py` патчит головы по стабильным
  атрибутам (cv2/cv3/nl/proto/one2one), метрики `ultralytics.utils.*` (ap_per_class, box_iou,
  non_max_suppression, polygons2masks_overlap) — стабильный API. Экспорт в фикс. контейнере
  (8.4.105): пока образ не пересобран с новой версией — ничего не сломается.

### Active
- **Мониторинг троттлинга в fps_bench (2026-08-11)**: во время measure-окна сэмплируются
  NPU-частота (`/sys/kernel/debug/rknpu/freq`), NPU-температура (thermal_zone6), максимум температур
  всех thermal-зон и частоты CPU per-core (`scaling_cur_freq`). В результат: `npu_temp_avg/min/max`,
  `npu_freq_avg/min`, `thermal_max_c`, `cpu_freq_ratio` (мин. avg/max по НАГРУЖЕННЫМ ядрам load>50% —
  иначе ложные срабатывания: в idle ядра легитимно снижают частоту). `throttled=True` + `throttle_reason`
  при: NPU freq <90% max **И загрузке NPU >50%** (недогрузка — не троттлинг!), cpu freq <90% max на
  нагруженном ядре, либо любая зона ≥80°C. `_bench_median`
  перед КАЖДЫМ повтором ждёт выхода из троттлинга (`_wait_no_throttle`, критерий — только температура,
  т.к. в idle частоты падают) с таймаутом `--throttle-timeout` (деф. 600с); при таймауте повтор/вариант
  НЕ запускается и помечается `error='throttle_timeout'`, fps=-1. В TSV добавлена колонка 'Троттлинг'
  (объединённые причины по вариантам). Замеры fps стали устойчивы к разогреву после convert/eval.
- **Фикс склейки JSON fps_bench (2026-08-11)**: rknnlite пишет в stdout без trailing \n
  (I RKNN: ...), JSON-результат приклеивался к RKNN-строке → benchmark.py не находил строку
  с `{` → fps записывались как -1. Фикс: `run_fps` ищет `rfind('{"model"')` + `raw_decode`,
  fps_bench печатает `'\n'+json+\n'` + flush. Результаты yolo26n (повторы=3, conf=0.25):
  cpu=5.08, fp16=13.0 (NPU 17.3%/CPU 23.4%), int8=17.69 (15.0/26.5%), int8-calib=17.68 (14.4/27.1%).
- **Resume-логика convert (2026-08-10)**: `benchmark/benchmark.py --phase convert` проверяет ВСЕ нужные
  варианты (onnx/fp16/int8/int8-calib), а не только `_fp16.rknn` — отсутствующий int8 теперь ДОСОБИРАЕТСЯ
  через `convert_bench.py`, а не тихо пропускается. `convert_bench.py` переведён на кеширующий
  `convert.run_convert()` (не удалённый `export_and_compile_all`): при полном кеше прогон занимает ~4с.
  `_build_rknn` собирает во ВРЕМЕННУЮ папку (`tempfile.mkdtemp`), а не рядом с onnx — иначе при onnx в кеше
  путь совпадал с целевым rknn и `copy2(src, src)` падал 'same file'; заодно при сбое сборки в кеше не
  остаётся обрубленного rknn.
- **Инвентаризация real-time моделей (2026-08-08)**: критерий — **самая лёгкая модель из каждой
  линейки** (не обязательно nano): добавлены **yolov3-tinyu** (23MB, самая лёгкая в yolo3; nl=2) и
  **yolov9t** (самая лёгкая в yolo9) — обе голова `Detect`, RK-opt патч подойдёт без изменений.
  **yolov10n** — после отдельного патча v10Detect (голова v10Detect end2end=True экспортируется корректно:
  `convert.py` ставит `m.end2end=False` всем модулям → fuse() не удаляет cv2/cv3; RK-opt совпал со std).
  **НЕ включаются**: RT-DETR (трансформер, NPU RK3588 не тянет attention), YOLOE/YOLO-World (open-vocab),
  YOLO-NAS (super_gradients не собрался, легаси). Ultralytics НЕ обновлялась (8.4.105 → 8.4.116 новых
  моделей нет). Детали — PLAN.md Этап G2.
- **Все 5 шаблонов синхронизированы и проверены** (detect/obb/pose/segment/classify):
  каждый обрабатывает ОБА формата (std single-output + RK-opt per-stride); `_generate_postprocess`
  генерирует все 5 без плейсхолдеров, py_compile OK; info из .pt корректна (nm=32, num_kpts=17,
  obb input 1024 → grid [128,64,32]).

### Next Move
1. **Полный бенчмарк yolo26n-det** — планируется: `cd benchmark && python3 benchmark.py --phase all --model yolo26n`.
   Полный eval **fp16** (5000, NPU) уже завершён напрямую eval.py (42 мин): map50=0.568, map50_95=0.408
   (эталон на onnx-вариант benchmark.py возьмёт из eval.py onnx/cpu). Осталось прогнать
   onnx(cpu)/int8/int8-calib варианты + validate/bench фазы. Оценка: ~3-4ч суммарно (все варианты
   с параллельным NMS по ~45 мин каждый).
2. **Interleaved runner**: smoke-тесты всех 6 задач ПРОЙДЕНЫ (col1==col2==col3, diverged=0).
   Полные прогоны (limit 500/5000) — по желанию пользователя.
3. **YOLO26/YOLO12/YOLO11 3-way ЗАВЕРШЕНЫ** — см. PLAN.md. RKNN-деплой (int8) — отложен по решению пользователя.
4. **obb на нативный 1024 — ЗАВЕРШЁН**: эталон всегда ultra .val на std ONNX (task='obb'); колонки col2/col3
   байт-в-байт; не-нативные артефакты удалены.
5. **Регрессия конвертера** (mini-метрики на конвертерных файлах) — по желанию пользователя.

### Relevant Files
- `PLAN.md` — план/статус всей работы (источник истины)
- `convert/templates/detect_postprocess.py` — multi-label + class-aware NMS + EVAL_PRECISE, оба формата (std + per-stride)
- `convert/templates/obb_postprocess.py` — fast_nms (batch_probiou), multi-label, оба формата
- `convert/templates/pose_postprocess.py` — kpts `(offset*2.0+grid-0.5)*stride`, sigmoid(cls/kpts), оба формата
- `convert/templates/segment_postprocess.py` — process_mask_160 (без sigmoid, `>0`≡`sigmoid>0.5`), clip всегда, оба формата (std=2 выхода, RK-opt=10 с proto в конце)
- `convert/templates/classify_postprocess.py` — PIL Resize(224)+CenterCrop(224)+ToTensor, top-1 raw score
- `benchmark/run_3way_eval.py` — interleaved-режим: один цикл по кадрам (col1 nms / col2 std / col3 rko), `compare_preds` с допусками, инкрементальные tp-агрегаты, diverged-список; результат в `cache/3way_results/<model>-<task>_interleaved.json`; флаги `--limit/--force/--save-diverged/--diverged-dir`
- `benchmark/export_std_onnx.py` — штатный экспорт std ONNX на нативном imgsz (без override) + перегенерация postprocess
- `percept/inference/__init__.py`: `/255.0` moved from `infer()` to default letterbox path only
- `benchmark/eval.py`: MIN_WH=1/MAX_DET=300/DEDUP_ENABLED=False в eval mode; `compute_inmemory_box_map`, `compute_inmemory_pose_map` (GT=orig_px*scale+pad), `compute_inmemory_obb_map` + `xyxyxyxy2xywhr_np`, `compute_inmemory_seg_map` (polygons2masks_overlap + декод + gt_cls[order]); EVAL_DUMP_PREDS→`benchmark/dump_preds.pkl`; ветка detect/pose/segment собирает preds_by_img/scales_pads; **per-image tp-функции `_box/_pose/_seg/_obb_tp_per_image` + `_ap_finalize(prefix=...)`, `_seg_finalize` — обёртка над `_ap_finalize` (2026-08-08)**; `load_yolo_as_coco` возвращает 4 значения (`images, ann_by_img, cat_ids, tmp.name`); **parallel NMS конвейер в detect-ветке (2026-08-10)**: `_detect_pre_nms` + `_detect_nms_finish` + `_big_core_count()`, NMS в ThreadPool по числу "больших" ядер, infer последовательно — полный fp16 eval yolo26n 5000 кадров за 42 мин (2.7x)
- `benchmark/prof_one_frame.py` — поэтапный профиль 1 кадра (letterbox/pre-NMS/NMS/finish) + сверка конвейерной пары == шаблонный postprocess (OK); `benchmark/prof_nms.py` — torchvision vs ultralytics NMS на 30000 кандидатов; `benchmark/prof_nms_parallel.py` — ThreadPool (4) vs ProcessPool (8) vs sequential; `benchmark/prof_batch_infer.py` — RKNN батч-8 vs bs1×8 на реальном NPU (1.38x)
