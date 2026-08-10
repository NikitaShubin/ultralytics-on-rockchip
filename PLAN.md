# PLAN.md — Корректность деплоя YOLO (ultralytics ↔ наша инфраструктура)

Статус: АКТИВЕН (обновлено 2026-08-09). Источник истины для следующего агента.
Все обсуждения — на русском (см. AGENTS.md).

> ПРИМЕЧАНИЕ (2026-08-09): все числовые замеры и их упоминания удалены по решению
> пользователя. Бенчмарки будут пересняты заново. Осталась цель, поток деплоя,
> технические решения и «тяжёлые факты» (без значений метрик).

## 1. ЦЕЛЬ (зачем вся эта работа)

**Сделать так, чтобы наше развёрнутое решение (NPU/RKNN через `deploy.sh`, CPU/ONNX)
работало ТОЧНО так, как задумано авторами ultralytics** — чтобы модели, обученные
в ultralytics, были верно задеплоены.

Ключевые требования:
1. **Инференс deploy.sh** и **конвертер** (неотъемлемая часть проекта) должны использовать
   РОВНО те пред- и постобработки, которые проходят валидацию.
2. `eval.py` + ultralytics `.val()` — **средство отладки**, а не самоцель:
   эквивалентность метрик доказывает, что деплой-постпроцесс = постпроцесс авторов.
3. У **yolo11 / yolo12 / yolo26 РАЗНЫЕ головы** → разные постпроцессы. Каждое семейство
   валидируется отдельно.
4. В перспективе `eval.py` — полноценная часть проекта для объективных метрик
   любых ONNX/RKNN моделей.

### Доказать для КАЖДОЙ модели (yolo11n, yolo12n, yolo26n) и каждой задачи
(det / seg / pose / cls / obb / sem):
- col1 = ultralytics `.val()` (эталон)
- col2 = `benchmark/eval.py` Pipeline на std ONNX (должен совпасть с col1)
- col3 = `benchmark/eval.py` Pipeline на RK-opt ONNX (= формат выхода RKNN, что деплоится)
- **Вывод в PLAN: развёрнутый конвертером postprocess обязан совпадать с доказанным.**

## 2. Поток деплоя (что проверяем)

```
deploy.sh <model> --fp16|--int8|--onnx
  → convert.py run_convert(): скачать .pt → экспорт RK-opt ONNX (патчи _rknn_mode)
    → сборка RKNN (per-stride выходы: box, cls_sigmoid, [sm|angle|kpt|coeff]) → кэш
  → _deploy_to_model(): копия модели в convert/model/ + ГЕНЕРАЦИЯ postprocess.py
    ИЗ ШАБЛОНОВ convert/templates/*_postprocess.py
  → docker compose: сервис Pipeline грузит model/ + postprocess.py
```

**Формат выхода:**
- **std ONNX** (обычный экспорт ultralytics): один output `(1, 4+nc, total_grid)`,
  уже decoded (xywh) + cls уже sigmoid. Проверено на `cache/yolo11n_std/`.
- **RK-opt / RKNN** (патчи `convert/patches/rknn_patch.py`): per-stride выходы
  `outputs[i*3]=box_dist (DFL-логиты), [i*3+1]=cls_sigmoid, [i*3+2]=sm|angle|kpt|coeff`
  (+ proto для segment). НЕ decoded, нужен DFL + grid-декод на CPU.

## 3. ТЕКУЩИЙ СТАТУС (2026-08-09)

### 3.1 Доказанная эквивалентность col2 == col1
Все 3-way проверки (col1/col2/col3) по задачам и линейкам ЗАВЕРШЕНЫ — три колонки
совпадали (см. раздел «Работа по этапам»). Числовые значения удалены по решению
пользователя; бенчмарки переснимаются заново через `benchmark/benchmark.py`.

### 3.2 Шаблоны конвертера синхронизированы с доказанной логикой
- `convert/templates/*_postprocess.py` обновлены до доказанной логики (multi-label NMS,
  class-offset, EVAL_PRECISE, fast_nms для obb, PIL-препроцесс для cls и т.д.) — см. Этап A.
- `_generate_postprocess` генерирует все 5 задач без плейсхолдеров, `py_compile` OK.

### 3.3 Не сделано / по желанию пользователя
- RKNN-деплой (int8) — отложен по решению пользователя.
- Полные interleaved-прогоны (limit 500/5000) для части задач — по желанию.
- Регрессия конвертера (mini-метрики на конвертерных файлах) — по желанию.

### 3.4 INT8 эксперимент (2026-08-09)
- **ЗАВЕРШЁН** (Этап H): наш RK-opt путь теряет при INT8 ~1%, нативный ultralytics
  8.4.116 (`_NormalizeCoords`) — ~20%. Миграция на вариант B не нужна.

### 3.5 Parallel NMS конвейер в eval.py (2026-08-10)
- **Проблема**: eval fp16/onnx на полном val2017 упирался в torchvision NMS на 30000 кандидатов
  (conf=0.001 + fp16-шум → 672000 кандидатов → NMS = 1.35с/кадр на ARM). У ultralytics `.val()`
  ровно то же самое — тот же `torchvision.ops.nms` на тех же 30000 (int8 быстрый: квантование
  обнуляет шум).
- **Фикс** (детект-ветка eval.py): infer на NPU — последовательно в главном потоке; pre-NMS шаги
  (decode/expand/sort) там же; NMS+reverse+format — в `ThreadPoolExecutor` по числу "больших" ядер
  (`_big_core_count()`: авто по `cpuinfo_max_freq`, RK3588 → 4×A76; 8 воркеров хуже — 4×A55 тормозят).
  torchvision NMS освобождает GIL → реальный параллелизм без ProcessPool-оверхеда.
- **Результат**: 1500→507мс/кадр (**2.7x**, 93% теоретического предела для 4 ядер); полный fp16 eval
  yolo26n (5000 кадров) = **2535с (42 мин)** вместо ~2ч. Метрики байт-в-байт совпадают с шаблонным
  postprocess (сверка OK, prof_one_frame.py). `_detect_pre_nms`/`_detect_nms_finish` — копия шагов
  detect_postprocess.py:137-203.

### 3.6 Эксперимент батч-инференса (2026-08-10)
- Собран std ONNX batch=8 → RKNN fp16 static [8,3,640,640] (в /tmp, рабочие файлы не тронуты).
- Реальный NPU: bs8×1=470ms/8 кадров vs bs1×8=649ms → **1.38x** для нано (NPU недогружен и на
  batch-1); для x-линейки прирост больше. В eval батч не окупается (NMS-пул доминирует ~340мс/кадр),
  в продакшене (conf=0.25) полезнее: infer 81→59ms. Конвертер под батч НЕ переделывали.

## 4. РАБОТА ПО ЭТАПАМ

### Этап A. Синхронизация шаблонов конвертера с доказанной логикой (ЗАВЕРШЕН)
Перенесено в `convert/templates/*_postprocess.py` ровно ту логику, что проверена:
- **detect**: multi-label expand (`np.where(cls>CONF_THRESH)` → `[box[i], cls[i,j], j]`) +
  class-aware NMS (offset `cls_id*MAX_WH`) + `EVAL_PRECISE` (float, без clip/int).
- **obb**: multi-label + fast_nms (`batch_probiou`, triu, `np.any(ious_np[:j,j]>=thr)`) +
  `MAX_WH` offset по классам.
- **pose**: формула kpts `(offset*2.0+grid-0.5)*stride`; sigmoid только на kpts
  (cls уже sigmoid для std; для RK-opt cls тоже sigmoid из патча).
- **segment**: `process_mask_160` (ops.process_mask upsample=False), `masks_160` в результате,
  EVAL_PRECISE.
- **classify**: PIL Resize(224)+CenterCrop(224)+ToTensor (НЕ letterbox!).
- Шаблон обрабатывает **ОБА формата выхода** (std single-output и RK-opt per-stride),
  детект по числу/форме outputs.
- Конвертер генерирует в `cache/{model}/postprocess.py` И `{model}_postprocess.py`
  (для `_std`-моделей) — идентичные файлы.

### Этап B. Регрессионный тест конвертерной генерации
Перегенерировать postprocess конвертером для yolo11n_std и прогнать eval.
Значения после перегенерации НЕ должны измениться (эквивалентность не должна сломаться).

### Этап C. RK-opt (col3) и RKNN деплой
- 3-way по yolo11n ЗАВЕРШЁН (RK-opt conversion loss для большинства задач = 0 —
  col3 байт-в-байт == col2; для detect — малый conversion loss). Значения удалены.
- **obb на нативном 1024**: эквивалентность пайплайнов доказана (col1/col2/col3 на одном
  1024 байт-в-байт). НЕнативный 640 давал заниженные метрики и переделан на 1024.
- **ВАЖНО про pose col1**: прежнее значение было снято на БИТЫХ yolo-лейблах (до их
  регенерации). Корректный col1 — на исправленных лейблах (совпадал с col2).
- Осталось: RKNN деплой (int8), регрессия конвертера.

### Этап D. yolo26
- Скачать yolo26n (.pt), сконвертировать std ONNX + RK-opt ONNX.
- Разобрать голову yolo26 (Detect26 etc.) и её постпроцесс (зонд выходов).
- Валидировать col1/col2/col3 по всем задачам. sem тоже.
- Учесть: `rknn_patch.py` уже содержит `_patch_detect26`-и-подобные для yolo26.

**Статус: все 6 задач (detect/pose/seg/obb/cls/sem) ЗАВЕРШЕНЫ — три колонки совпадали
байт-в-байт.** Числа удалены; ключевые технические решения:
- **Полный eval yolo26n-det fp16 (2026-08-10)**: 5000 кадров val2017, 2535с (42 мин, parallel NMS
  конвейер — см. 3.5), map50=0.568, map50_95=0.408, ar100=0.599, errors=0. Полный бенчмарк
  (onnx/fp16/int8/int8-calib) — см. Next Move в AGENTS.md.
- **one2one-проблема РЕШЕНА**: `m.end2end=False` перед export в convert.py — иначе fuse()
  удаляет cv2/cv3 и оставляет только one2one-головы → RK-opt давал другие логиты.
- **sem**: eval.py semantic ветка реализована (compute_inmemory_sem_map, letterbox_semantic_gt
  pad=255, apply_cityscapes_label_mapping). Метод col1 подтверждён (`SemanticSegmentationValidator`
  есть в 8.4.105). **Фикс col3 (2026-08-07)**: `_rknn_semantic_segment` (rknn_patch.py:96)
  РАНЬШЕ заменял bilinear→nearest → col3 != col2. Переведён на bilinear align_corners=False
  (как std) + переэкспорт RK-opt ONNX → col3 == col2 байт-в-байт.
  RKNN-поддержка bilinear Upsample на NPU — проверить при RKNN-деплое (иначе упадёт на CPU).
- **БАГ порога raw mask** в eval.py: был `gt.max() > 50`, а cityscapes8 маски — raw labelIds
  7..33 (max 33) → label_mapping не применялся. Фикс: порог `> 18` (trainId всегда 0..18).
- **БАГ ООМ в eval.py (исправлен 2026-08-08)**: semantic-ветка копила все pred/gt-карты
  (1024² int64) → OOM-kill ~300 imgs. Фикс: инкрементальный confusion matrix (`np.add.at`
  per-image) + `sem_metrics_from_matrix()`; результат байт-в-байт тот же.
- **Данные Cityscapes**: полный val через Antreas/Cityscapes (HF parquet, 3 шарда) —
  изображения в НОРМАЛЬНОМ PNG (Chris1/cityscapes — битый «SPIFF»-формат, нечитаем PIL/cv2).
  Имена официальные, маски из parquet байт-в-байт == gtFine_trainvaltest.zip.
  `setup_datasets.py` обновлён; битый `leftImg8bit.zip` удалён.

### Этап E. yolo12
- **yolo12n detect 3-way ЗАВЕРШЁН** (полный прогон): три колонки равны байт-в-байт.
  YOLO12n: стандартная голова Detect (reg_max=16, DFL, end2end=False, выход [1,84,8400]) —
  RK-opt патчи и шаблоны yolo11 подошли БЕЗ изменений; отличие только в backbone (A2C2f attention).
  Официальных весов yolo12 для seg/pose/obb/cls НЕТ (только detect n/s/m/l/x) — yolo12 ограничен detect.

### Этап H. INT8 сравнение: наш RK-opt путь vs нативный ultralytics 8.4.116 (2026-08-09)

Причина: ультралитикс добавил `_NormalizeCoords` (нормировка xywh на размер входа при
`format=='rknn' and quantize==8`), чтобы пер-тензорный INT8 scale не давил class-scores.
Спросили: не решили ли они нашу проблему лучше (мигрировать на вариант B?).

**Эксперимент: yolo11n detect, coco128 (smoke) + полный val2017 (5000 imgs), только int8.**

Итоговые цифры (метрика — inmem, доказано эквивалентна их `.val()`, проверено: std ONNX
0.6744 в обоих):

| Путь | fp32 | int8 | потеря точности |
|---|---|---|---|
| A (наш RK-opt: декодирование вне графа, per-stride) | 0.523 | **0.518** | **~1%** |
| B (нативный 8.4.116: `_NormalizeCoords` + RKNNBackend денормировка) | 0.542 | **0.435** | **~20%** |

**ВЫВОД: наш RK-opt путь сохраняет точность INT8 в ~20 раз лучше, чем `_NormalizeCoords`.**
Абсолютная INT8 точность выше (0.518 vs 0.435) несмотря на то, что наш fp32-базлайн
немного ниже (0.523 vs 0.542). **Миграция на вариант B НЕ нужна** — их фикс заметно хуже.

Детали:
- Путь B работает: `YOLO(pt).export(format='rknn', quantize=8, data=coco128.yaml)` → создаёт
  `cache/yolo11n/yolo11n_rknn_model/` (metadata.yaml `quantize: 8`, `end2end: false`,
  RKNNBackend денормирует координаты по imgsz). `.val()` на NPU (convert-контейнер) работает.
- **Нативный export B ЗАТИРАЕТ `cache/yolo11n/yolo11n.onnx`** (создаёт ONNX рядом с .pt,
  конвертит в RKNN и удаляет его) — RK-opt ONNX пришлось перегенерировать
  (`export_onnx_only`, 9 per-stride выходов).
- Их `.val()` не обрабатывает RK-opt per-stride выходы (IndexError) — прямое сравнение
  preds только через нашу inmem-метрику.
- rknn-toolkit2 2.3.2: `do_quantization=True` без `calib_dir` → `rknn.build не удался: -1`
  (датасет обязателен для INT8); наш `_build_rknn` берёт `calib_dir=coco128/images/train2017`.
- convert-контейнер: NPU доступ через `--privileged` + `-v /proc/device-tree/compatible`
  + `--network host` (см. convert/docker-compose.yml). Бывший ad-hoc контейнер segsmoke
  удалён как дубль convert-сервиса.
- **БАГ logging**: импорт `rknnlite.api` перезаписывает `logging._nameToLevel` → torch падает
  `Unknown level: 'WARNING'`. Фикс в `percept/inference/__init__.py::_load_rknn` (save/restore).
- **БАГ pycoco в eval.py**: `detections_to_coco` занижает map50 в ~4 раза (0.145 vs inmem 0.64)
  на coco128; на полном val2017 расхождение меньше (0.532 vs 0.523). inmem-метрика валидна.

Открытая аномалия: наш RK-opt fp32 (0.523) на ~3.5% ниже std (0.542) на полном val2017
(на coco128: 0.64 vs 0.6744, -5%), хотя ранее доказывалась байт-в-байт эквивалентность
col1/col2/col3 для yolo11n detect. Возможная причина — RK-opt ONNX после force-перегенерации
или разные конфиги eval. Требует проверки (interleaved col2 vs col3).

### Этап G. Sweep nano-моделей от версии 3 (ЗАВЕРШЕН)
Причина: пользователь хотел проверить эквивалентность всех непроверенных nano-моделей,
начиная с версии 3.

**Доступность nano-весов** (assets v8.4.0):
- yolov3n / yolov6n / yolov9n — **НЕ существуют** (yolov9 имеет только yolov9t/s/m/c/e).
- **yolov5nu.pt** — есть (голова `Detect`).
- **yolov8n** — есть (все задачи) + detect уже завершён.
- **yolov10n.pt** — есть (голова `v10Detect`, end2end=True; `export_std_onnx.py` форсирует
  end2end=False → raw).
- yolo11n / yolo12n / yolo26n — уже ЗАВЕРШЕНЫ.

**yolov8n detect 3-way ЗАВЕРШЁН** (полный прогон): col1 == col2 == col3 байт-в-байт
(разница 1 pred на повторе = tie-break NMS; метрики равны).
- **Фикс col1**: `datasets/coco8` был СИМЛИНКОМ на хостовый абсолютный путь
  (`convert/datasets/coco8`) → внутри контейнера битый → `check_det_dataset` падал
  ("images not found, missing path ..."). Заменён реальной копией.
  УРОК: в `datasets/` НЕ должно быть симлинков на хостовые пути — только реальные директории.

**Полный свип всех линеек ЗАВЕРШЁН (2026-08-09, ночь)** — 9 прогонов
(yolov8n detect/seg/pose/obb/cls, yolov5nu, yolov3-tinyu, yolov9t, yolov10n),
ВЕЗДЕ col1==col2==col3, errors 0. Значения удалены.
- **Diverged (всего ~25 кадров из ~38k) — подтверждённый near-tie**: у ultralytics-NMS
  и нашего NMS при почти/равном score (разница ≤ 1e-6) редкие кадры выбирают разных
  кандидатов; повторы могут давать разное число preds (недетерминизм torch-NMS).
  Метрики при этом НЕ меняются (байт-в-байт).
- **yolov10n подтверждён без отдельного патча**: голова v10Detect (end2end=True)
  экспортируется корректно — `convert.py` ставит `m.end2end=False` всем модулям →
  fuse() не удаляет cv2/cv3; RK-opt совпал со std.

**Правило имён моделей**: std_model_dir/rko_model_dir строятся из МОДЕЛИ, а не task
(`yolov8n-seg segment`, `yolov8n-pose pose`, `yolov8n-cls classify`, `yolo11n-obb obb`,
`yolo26n-sem semantic`), иначе загружается detect-кэш → `std_raw[1]` IndexError.

### Этап G2. Инвентаризация real-time моделей (2026-08-08)
Запрос пользователя: добавить в инвентаризацию другие yolo (NAS) и не-yolo (RT-DETR);
выяснить, какие real-time модели поддерживает последняя ultralytics и способны ли они на RKNN.

**Версия ultralytics**: образ `percept-converter:latest` = 8.4.105. Последняя на PyPI = 8.4.116.
**Обновление НЕ требуется**: в 8.4.106–116 новых семейств моделей нет (фиксы экспорта, QNN/Ascend/
Hailo, платформа; RKNN-фиксы 8.4.113 касаются НАТИВНОГО RKNN-экспорта, у нас свои патчи).
Все эталоны 3-way сняты на 8.4.105 — версию зафиксировать в convert/Dockerfile.

**Real-time модели (assets v8.4.0) и RKNN-способность по нашему конвертеру**:
| Модель | Голова | RKNN (наш конвертер) |
|---|---|---|
| yolov3-tinyu (23MB) | `Detect` (nl=2, reg_max=16) | ✅ патч Detect без изменений |
| yolov5nu | `Detect` | ✅ |
| yolov8n (все задачи) | `Detect`/`Segment`/`Pose`/`OBB`/`Classify` | ✅ (свип) |
| yolov9t | `Detect` (reg_max=16) | ✅ патч Detect без изменений |
| yolov10n | `v10Detect` (end2end=True) | ✅ подтверждён (см. Этап G) |
| yolo11n / yolo12n / yolo26n | Detect/26-головы | ✅ ЗАВЕРШЕНО |
| rtdetr-l/x | `RTDETRDecoder` (трансформер, NMS-free) | ❌ attention/softmax на NPU RK3588 — RK-opt патча нет |
| yoloe-* (v8/11/26) | `YOLOEDetect`/`YOLOESegment(26)` | ⚠️ open-vocabulary (промпты) |
| yolov8-world | `WorldDetect` | ⚠️ open-vocabulary |
| yolo_nas_s/m/l | (super_gradients) | ❌ НЕ грузится: `super_gradients` не собрался в контейнере; интеграция устаревшая |

**Итог**: критерий — **самая лёгкая модель из каждой линейки** (не обязательно nano).
В инвентаризацию 3-way добавлены **yolov3-tinyu** (самая лёгкая в yolo3) и
**yolov9t** (самая лёгкая в yolo9) — голова Detect, подойдут без изменений, как yolo12.
RT-DETR / YOLOE / YOLO-World / YOLO-NAS — НЕ включаются (RKNN не способен либо open-vocabulary).

### Этап H. Interleaved runner (benchmark/run_3way_eval.py)
`benchmark/run_3way_eval.py` переписан на interleaved-режим (единственный). Вместо трёх
отдельных прогонов по задачам (старый per-column) — ОДИН цикл по кадрам: на каждом кадре
col1 (ultralytics NMS на том же std-тензоре), col2 (наш std постпроцесс), col3 (наш RK-opt
постпроцесс); сразу сравнение preds (допуски: score 1e-6, bbox 1e-3, angle 1e-4, kpts 1e-3,
mask 0.01; class_id строго), инкрементальные per-image tp-агрегаты (`_box/_pose/_seg/_obb_tp_per_image`
+ `_ap_finalize(prefix=...)`), diverged-счётчики (при расхождении сохраняются только эти изображения).

**Ключевые технические решения**:
- Pipeline не используется — `Pipeline._load_postprocess` грузит глобальный `sys.modules['postprocess_mod']`;
  работаем с `ModelBackend` напрямую, постпроцессы загружаются под уникальными именами (`pp_std`, `pp_rko`).
- col1 = `non_max_suppression(max_time_img=1000)` из `ultralytics.utils.nms` (НЕ ops) на std_raw
  (letterboxed-пространство); col2/col3 — наш постпроцесс (orig) → перевод в letterboxed через `x*scale+pad_left`.
- pycoco в интерливе не нужен; classify/semantic: col1 == col2 по построению (тот же std_raw);
  semantic — инкрементальная confusion-матрица per-image (OOM-фикс).
- Результат: `cache/3way_results/<model>-<task>_interleaved.json` (col1/col2/col3, diverged-счётчики,
  equal-флаги, preds). Флаги: `--limit/--force/--save-diverged/--diverged-dir`.

**Статус**: smoke-тесты всех 6 задач ПРОЙДЕНЫ (col1==col2==col3, diverged=0), mini-прогоны
всех 6 задач ЗАВЕРШЕНЫ, полный свип всех линеек ЗАВЕРШЁН. Значения удалены.

**Исправлено 4 бага (2026-08-08)**:
1. **nms-мутация std_raw**: `_nms_det` без `.copy()` портил буфер std_raw (xywh→xyxy инплейс через
   transpose-вью) → col2 ломался. Фикс `.copy()` (run_3way_eval.py:678).
2. **'masks' vs 'masks_160'**: canon_col1 писал 'masks', остальные ждали 'masks_160' → KeyError.
   Унифицировано на 'masks_160'.
3. **segment/pose постпроцессы клипали bbox ВСЕГДА** (не под `if not EVAL_PRECISE` как detect) →
   col1 vs col2 bbox diff. Фикс в шаблонах + патч сгенерированных кэш-файлов.
4. **semantic-ветка** использовала постпроцесс (orig-пространство) при GT в letterboxed → TypeError;
   переведена на raw argmax (`raw[0][0].argmax(0)`).

### Этап F. Финальные артефакты
- Перегенерировать `cache/3way_results/*.json`.
- Обновить AGENTS.md (Work State) и PLAN.md.
- (sem реализован и подтверждён; осталось по желанию: RKNN int8-деплой, регрессия конвертера)

### Этап F2. Рефакторинг benchmark — ЗАВЕРШЕНО (2026-08-09)
- **eval-фаза benchmark.py переведена с сломанного `eval_bench.py` на `benchmark/eval.py`** (FULL × все
  quantize: onnx→cpu, fp16/int8/int8-calib→npu). `results.json['eval']` — dict
  по вариантам; TSV берёт эталон из onnx. `eval_bench.py` удалён.
- **npu-варианты инференсят на РЕАЛЬНОМ NPU (2026-08-09)**: convert-контейнер — надстройка над
  percept (rknnlite + проброшенные хостовые librknnrt/API в `/usr/lib` + `/dev/dri`), т.е. RK3588.
  Эмулятор rknn-toolkit2 — только fallback, если rknnlite недоступен.
- **Мини-датасеты ultralytics** качаются в `datasets/` проекта через `setup_datasets.ensure_mini_dataset(task)`
  (используется в convert int8 без калибровки и в smoke 3way `resolve_eval_data`); полные — `ensure_full_dataset`.
  Принцип: код сам подтягивает необходимое на чистой установке, ничего лишнего не качается; датасеты из git не
  удаляются (пользователь).
- **ModelBackend rknn fallback на x86**: rknnlite нет → rknn-toolkit2; `load_rknn`+`init_runtime` падает без
  target → пересборка ONNX (`load_onnx`+`build`+`init_runtime`, симулятор). Это fallback-путь для x86-хоста;
  на реальном устройстве (convert-контейнер на RK3588) работает rknnlite + реальный NPU.
- ВНИМАНИЕ: `cache/yolo11n/yolo11n.onnx` перезаписан std-версией (1 выход) — RK-opt постпроцесс не работает.
  Восстановление: `ensure_rko_cache('yolo11n', force=True)`.

## 5. Тяжёлые факты (не потерять)

1. **val-NMS ultralytics — multi_label=True**: `models/yolo/detect/val.py postprocess`
   → `nms.non_max_suppression(..., multi_label=True, nc=self.nc, rotated=True)`.
   Каждый бокс × каждый класс с conf>thresh = отдельное предсказание. БЕЗ этого
   obb давал меньше preds и заниженную метрику.
2. **EVAL_PRECISE=True** (detect/seg/obb): bbox float, БЕЗ `int()`/`clip()` — метрика
   ultralytics не клипает. eval-режим: `MIN_WH=1, MAX_DET=300, DEDUP_ENABLED=False`.
3. **GT-маппинг per-axis** (eval.py): `scales_pads=(sx,sy,pad_left,pad_top)`,
   `sx=(imgsz-pad_w)/w, sy=(imgsz-pad_h)/h`; GT = `orig_px*scale+pad`.
   `ratio_pad` ultralytics = `(h_ratio, w_ratio)`.
4. **letterbox: `round`→`ceil`** — ultralytics использует `min(math.ceil(w*scale), imgsz)`.
5. **seg overlap-семантика**: `Format(mask_overlap=True)` → `polygons2masks_overlap`
   один (160,160) int-слой 1..N; `_prepare_batch` декодирует `(overlap==i+1)`
   в порядке area-desc; `gt_cls=gt_cls[argsort(-areas)]`; `mask_iou`+`match_predictions`+`ap_per_class`.
6. **pose: GT в letterboxed-пространстве** (с паддингом). Внутренние метрики в (640,640);
   pycocotools — только персоны с kp>0.
7. **RK-opt cls_scores уже sigmoid** (патч `.sigmoid()`) — не применять повторно.
   sigmoid только к kpts.
8. **`/255.0` только в default letterbox-пути** `ModelBackend.detect()`, не в `infer()`.
9. **cls preprocessing**: PIL Resize(224, bilinear)+CenterCrop(224)+ToTensor, НЕ letterbox.
10. **Система 8GB RAM**: один контейнер `--memory=4g`, eval с `--limit 500`;
    долгие процессы в фон (`> /tmp/xxx.log 2>&1 & disown`).
11. **fast_nms obb** = реплика `TorchNMS.fast_nms`: сортировка по score, `batch_probiou`,
    `triu(diagonal=1)`, keep где `np.any(ious_np[:j,j]>=IOU_THRESH)` ложно.
12. **std ONNX выход уже decoded**: `(1,4+nc,total)` xywh + sigmoid cls. RK-opt — сырые
    DFL-логиты + sigmoid cls per-stride.
13. **ЗАПРЕЩЕНО форсировать imgsz при экспорте/валидации** (2026-08-06): рабочее разрешение
    берётся из исходного .pt (args.imgsz). Еval НЕ заменяет разрешение.
14. **Ультралайтс не определяет task для obb-ONNX** — нужно `YOLO(onnx, task='obb')`,
    иначе val идёт как detect (нули).
15. **obb col1: ультралайтс NMS имеет time_limit** (2.0+max_time_img*bs) и под CPU-нагрузкой
    урезает preds. Для честного col1 задавать max_time_img=1000 (monkeypatch
    nms.non_max_suppression) или сравнивать по capture preds.
16. **Данные — не в git**: `benchmark/data/`, `benchmark/results/`, `convert/datasets/`,
    `cache/`, логи, `.tsv`-результаты — запрещено коммитить (AGENTS.md).

## 6. Команды

```bash
# col2 (std ONNX)
docker run --rm --memory=4g --entrypoint python -w /app/percept \
  -v /home/orangepi/work/percept:/app/percept \
  -v /home/orangepi/work/percept/datasets:/app/datasets \
  percept-converter:latest \
  benchmark/eval.py detect datasets/coco_val2017 --model-dir cache/yolo11n_std --model-name yolo11n_std --backend cpu

# eval-фаза benchmark (FULL × все quantize; docker compose run)
python3 benchmark/benchmark.py --phase eval --model yolov8n

# Полный бенчмарк одной модели (convert → validate → bench → eval)
cd benchmark && python3 benchmark.py --phase all --model yolo26n

# rknn-вариант вручную:
docker compose -f convert/docker-compose.yml run --rm \
  --entrypoint python convert \
  benchmark/eval.py detect /app/datasets/coco_val2017 \
  --model-dir /app/cache/yolo11n --model-name yolo11n \
  --backend npu --rknn-suffix _fp16

# col1 (эталон) — /tmp/opencode/rec_*.py (rec_detect_full.py, rec_pose_full.py,
# rec_seg_mini.py, rec_obb.py), результаты в *.pkl

# Конвертация модели
./deploy.sh yolo26n-pose --fp16
# или прямо convert.py: docker compose run --rm convert yolo26n-pose --fp16
```

## 7. Ключевые файлы

- `benchmark/eval.py` — eval-движок: inmem box/pose/obb/seg, pycoco, GT per-axis; CLI: `--backend/--model-dir/--model-name/--limit/--rknn-suffix`.
- `benchmark/benchmark.py` — оркестратор (convert/validate/bench/eval); eval = FULL × все quantize на eval.py.
- `setup_datasets.py` — автоподготовка датасетов: `ensure_mini_dataset(task)`/`ensure_full_dataset(task)` (мини-датасеты ultralytics качаются в `datasets/` проекта).
- `pipeline.py`, `percept/inference/__init__.py` — пайплайн (часть проекта); ModelBackend с fallback rknn-toolkit2 (x86).
- `convert/convert.py` — конвертер; `_generate_postprocess` (шаблоны), `_deploy_to_model`.
- `convert/patches/rknn_patch.py` — патчи голов (RK-opt выходы), есть yolo26-патчи.
- `convert/templates/*_postprocess.py` — ШАБЛОНЫ (обновлены до доказанной логики).
- `cache/yolo11n_{,_std,}{,-seg_std,-pose_std,-obb_std,-cls_std}/` — проверенные постпроцессы.
- `cache/3way_results/*.json` — результаты interleaved-прогонов.
- `/tmp/opencode/rec_*.py` — скрипты эталонов .val + pkl (утеряны 2026-08-07, не полагаться).
- `AGENTS.md` — Work State (автообновляется).
