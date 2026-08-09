# PLAN.md — Корректность деплоя YOLO (ultralytics ↔ наша инфраструктура)

Статус: АКТИВЕН (обновлено 2026-08-08). Источник истины для следующего агента.
Все обсуждения — на русском (см. AGENTS.md).

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

## 3. ТЕКУЩИЙ СТАТУС (2026-08-08, вечер)

### 3.1 Доказанная эквивалентность col2 == col1 для yolo11n

| задача | col1 | col2 (eval.py std ONNX) | статус |
|---|---|---|---|
| detect | 0.542 (5000) | inmem 0.5764/0.4237 mini 200 imgs; полный map50=0.532 (ONNX-conversion loss ~1.8%) | ✅ mini байт-в-байт (34782/34783 preds) |
| seg | mask 0.5382/0.3615 (mini 200) | inmem 0.5377/0.3596; pred-маски байтово идентичны | ✅ |
| pose | box 0.5841/0.4323, pose 0.4564/0.2851 (5000) | ровно те же значения | ✅ |
| cls | top1 0.700 (50000) | top1 0.70008 | ✅ |
| obb | map50 0.4769/0.3427, preds 82057 (458) | inmem 0.4769/0.3426, preds 82057 | ✅ preds байт-в-байт |

**ПРОВЕРЕНО вручную на cache-постпроцессах (hand-fix). Шаблоны конвертера НЕ синхронизированы.**

### 3.2 Разрыв (главная проблема)
- Доказанные постпроцессы — hand-fix в `cache/{model}_std/*_postprocess.py` и
  `cache/yolo11n/postprocess.py`.
- `convert/templates/*_postprocess.py` — СТАРЫЕ: single-label NMS (`max(axis=1)`+`argmax`),
  без class-offset, без `EVAL_PRECISE`, obb без multi-label/fast_nms.
- Развёрнутый конвертером postprocess даёт ДРУГИЕ предсказания, чем доказанный.
- **col3 (RK-opt/RKNN) НЕ отлажен**: `detect_col3.json` map50=0.3632 vs col1 0.542
  (колоссальный разрыв — не просто NMS-семантика; надо разбирать).

### 3.2b ПРОГРЕСС Этапа A (2026-08-06)
- **detect-шаблон** (`convert/templates/detect_postprocess.py`): std 0/6 mismatch
  (823 obj) vs cache-эталон; per-stride 950/950, max-дист. 9e-05px.
- **obb-шаблон** (`convert/templates/obb_postprocess.py`): std байт-в-байт (1690 obj, 0 mismatch);
  **per-stride эквивалентен torch-эталону** на RK-opt ONNX (1024): 1164/1164 геометрия+угол
  (центры, w/h, angle 100%), классы 1101/1164, exact-multiset 1092/1164 — остаток = float-шум
  (near-tie multi-label близнецов, как detect img 11197).
- **ВАЖНОЕ ОТКРЫТИЕ**: `cache/yolo11n-obb/model.onnx` (RK-opt) сконвертирован с **входом 1024×1024**,
  а `yolo11n-obb_std` — 640×640. → col3 obb 0.6114 > col1 0.4765 объясняется РАЗНЫМ разрешением,
  НЕ метрикой. Для сопоставимого col3 нужно переэкспортировать RK-opt obb с imgsz=640.
- **ИСПРАВЛЕНО**: убран ошибочный swap (bh>bw → w/h swap + angle-=pi/2) из per-stride obb decode
  (ultralytics НЕ свапает предсказания; swap был в GT-лейблах). До фикса углы отличались на ~67°.
- **pose-шаблон** (`convert/templates/pose_postprocess.py`): std path **127/127 байт-в-байт**
  vs cache-эталон (bbox+conf+cls+kpts); per-stride path **эквивалентен torch-эталону**
  (diff ≤3e-5 px на kpts, 3.6e-07 conf; exact-match на round-границе — артефакт теста;
  несколько NMS near-tie флипов в плотных толпах — как detect img 11197).
- **segment-шаблон** (`convert/templates/segment_postprocess.py`, переписан):
  - std path (2 выхода: output0 (1,4+nc+NM,8400) + proto (1,NM,160,160)):
    **854/854 байт-в-байт** vs cache-эталон (bbox+conf+cls+masks_160) на 6 imgs.
  - per-stride path (10 выходов: [box_dist, cls_sigmoid, coeff]×3 + proto в конце):
    **607/607 байт-в-байт** с std-путём (bbox 0.1px, score 4dec, class, masks_160).
  - ВАЖНО: clip bbox ВСЕГДА (как cache-эталон, НЕ как detect/pose EVAL_PRECISE).
    Для mask-метрики не влияет (masks_160 по letterboxed-bbox до reverse letterbox);
    `>0` ≡ `sigmoid>0.5` без сигмоида (байт-эквивалент ultralytics).
- **classify-шаблон** (`convert/templates/classify_postprocess.py`, переписан):
  preprocess = PIL Resize(224,bilinear)+CenterCrop(224)+ToTensor (ultralytics
  classify_transforms, НЕ letterbox); postprocess = top-1 raw score. Проверен:
  preprocess байт-идентичен cache-эталону, top-1 (класс+score) 0 mismatch на 10 imgs.
- **convert.py `_generate_postprocess`**: все 5 задач (detect/obb/pose/segment/classify)
  генерируются из шаблонов без оставшихся плейсхолдеров, `py_compile` OK.
  `nm`=32 для seg, `num_kpts`=17 для pose, grid_sizes=[80,40,20] при 640 — заполняются верно.

### 3.3 Не сделано
- yolo12 — **detect ЗАВЕРШЁН** (3-way, full 5000); seg/pose/obb/cls невозможны (нет официальных весов).
- yolo26 — все 6 задач ЗАВЕРШЕНЫ (detect/pose/seg/obb/cls/sem), см. Этап D.
- sem — реализован и 3-way подтверждён (cityscapes8 4 imgs); полный Cityscapes (500 val) требует регистрации.
- seg полный col2 (5000) — не запущен (mini 200 достаточно для эквивалентности).
- `cache/3way_results/*.json` — старые per-column JSON устарели (генерировались до фиксов).
  С 2026-08-08 результаты пишутся в `<model>-<task>_interleaved.json` (см. Этап H).

### 3.4 Устаревшие JSON в cache/3way_results/ (НЕ использовать как финальные)
- `detect_col2.json`: map50 0.5504 (до EVAL_PRECISE); актуально mini 0.5764.
- `obb_col2.json`: 75106 preds, 0.467 (до multi-label); актуально 82057 / 0.4768.
- `detect_col3.json`: 0.3632 — RK-opt не отлажен.
- `classify_col2/col3.json`: 0.61258 (до cls-препроцесса); актуально 0.70008.

## 4. РАБОТА ПО ЭТАПАМ

### Этап A. Синхронизация шаблонов конвертера с доказанной логикой (критично)
Перенести в `convert/templates/*_postprocess.py` ровно ту логику, что проверена:
- **detect**: multi-label expand (`np.where(cls>CONF_THRESH)` → `[box[i], cls[i,j], j]`) +
  class-aware NMS (offset `cls_id*MAX_WH`) + `EVAL_PRECISE` (float, без clip/int).
- **obb**: multi-label + fast_nms (`batch_probiou`, triu, `np.any(ious_np[:j,j]>=thr)`) +
  `MAX_WH` offset по классам.
- **pose**: формула kpts `(offset*2.0+grid-0.5)*stride`; sigmoid только на kpts
  (cls уже sigmoid для std; для RK-opt cls тоже sigmoid из патча).
- **segment**: `process_mask_160` (ops.process_mask upsample=False), `masks_160` в результате,
  EVAL_PRECISE.
- **classify**: PIL Resize(224)+CenterCrop(224)+ToTensor (НЕ letterbox!).
- Шаблон должен обрабатывать **ОБА формата выхода** (std single-output и RK-opt per-stride),
  детект по числу/форме outputs (как `has_single_output` в obb-шаблоне).
- Конвертер генерирует в `cache/{model}/postprocess.py` И `{model}_postprocess.py`
  (для `_std`-моделей) — идентичные файлы.

### Этап B. Регрессионный тест конвертерной генерации
Перегенерировать postprocess конвертером для yolo11n_std и прогнать eval:
- detect mini: 0.5764/0.4237 (34783 preds)
- obb: 0.4769/0.3426 (82057 preds)
- seg mini: 0.5377/0.3596
- pose 5000: 0.5841/0.4323/0.4564/0.2851
- cls 50000: 0.70008
Значения не должны измениться после перегенерации.

### Этап C. RK-opt (col3) и RKNN деплой
- **yolo11n 3-way ЗАВЕРШЁН (2026-08-06)** — см. таблицу ниже (раздел «Итог yolo11n»).
  Conversion loss RK-opt: detect ~0.0002 (0.532→0.5318), pose/seg/cls/obb = 0 (байт-в-байт).
- Осталось на этапе C: RKNN деплой (int8), регрессия конвертера (mini-метрики на конвертерных файлах).
- **seg col3 mini (RK-opt)** (200 imgs): `cache/yolo11n-seg/postprocess.py` перегенерирован из
  segment-шаблона; **inmem_seg 0.5375/0.3598** vs col2 0.5377/0.3596 (дельта 0.0002 =
  conversion loss; pred-маски байтово идентичны, 607/607 на смоуке).
- **pose col3 mini (RK-opt)** (200 imgs): `cache/yolo11n-pose/postprocess.py` перегенерирован из
  pose-шаблона; **col3 == col2 БАЙТ-В-БАЙТ**: pycoco map50 0.8519/map50_95 0.6153/ar100 0.5882,
  inmem_box 0.6099/0.4553, inmem_pose 0.4893/0.296, 6112 preds (одинаково для std и RK-opt).
  RK-opt pose conversion loss = 0. Смоук 126/127 байт-эквивалент vs std.
- **cls col3 (RK-opt)** (5000 imgs = первые 100 классов ImageNet): **top1 0.7598 == col2 std 0.7598**
  на том же сабсете (RK-opt cls conversion loss = 0). 0.7598 vs col1 0.700 — разница из-за
  класс-смещения сабсета, НЕ модель. `cache/yolo11n-cls/postprocess.py` перегенерирован из шаблона.
- **obb col3 ПОЛНЫЙ (RK-opt, 640×640)** (2026-08-06): RK-opt ONNX переэкспортирован с
  imgsz=640 (бэкап 1024: `/tmp/opencode/yolo11n-obb_rkopt_1024.onnx`); `cache/yolo11n-obb/postprocess.py`
  перегенерирован из obb-шаблона (GRID [80,40,20], multi-label + class-offset fast_nms + float bbox).
  **map50 0.4768 / map50_95 0.3417 / ar100 0.5785, inmem 0.4769/0.3426, 82057 preds (458 imgs)** —
  байт-в-байт равно col2 std. RK-opt obb conversion loss = 0. Бэкап старого постпроцесса:
  `/tmp/opencode/yolo11n_obb_postprocess_old.py`.
- **ВАЖНО (2026-08-06, вечер)**: ВЕСЬ obb 3-way снят на НЕнативном 640 (нативный = 1024, .pt args.imgsz).
  Эквивалентность пайплайнов доказана корректно (col1/col2/col3 на одном 640, байт-в-байт 82057 preds),
  НО метрика занижена (RK-opt 1024 давал 0.6114 — НО это было со старым single-label постпроцессом;
  корректный эталон col1 на 1024 = 0.5758, см. ниже). Источник: ad-hoc `export_obb640.py`
  (`model.args["imgsz"]=640`) + `run_3way_eval.py` хардкодил imgsz=640. СКРИПТЫ ИСПРАВЛЕНЫ:
  `run_3way_eval.py` (resolve_imgsz из .pt, пути от ROOT), новый `benchmark/export_std_onnx.py`.
  **ПЕРЕДЕЛКА (2026-08-07, ЗАВЕРШЕНА ✅)**: RK-opt 1024 (9 per-stride выходов) и std 1024 готовы, оба
  postprocess GRID [128,64,32]; мусорный верхнеуровневый `cache/yolo11n-obb.onnx` (640) удалён.
  **ОБЁ obb 3-way на 1024 (458 imgs) ЗАВЕРШЁН**:
  - col1 = **0.5885/0.4444** (ultra .val на std ONNX, task='obb'! Без task=obb ультралайтс валидирует
    ONNX как detect → нули. На .pt col1 = 0.5758/0.4355 — систематика .pt vs ONNX ~0.012, НЕ эталон)
  - col2 (std ONNX) = **0.588/0.4447**, inmem 0.5885/0.4451, ar100 0.6766, 78981 preds
  - col3 (RK-opt) = **0.588/0.4447** — байт-в-байт == col2 (78981 preds) ✅ conversion loss = 0
  - Дельта col1↔col2 ~0.0005 — эквивалентность на нативном 1024 подтверждена (как на 640).
- **ИТОГ yolo11n 3-way (2026-08-06/07)**:
  | задача | col1 (ultra .val) | col2 (std ONNX) | col3 (RK-opt) |
  |---|---|---|---|
  | detect | 0.542 (full) | 0.532 (full) | 0.5318 (full) ✅ |
  | pose | box 0.584, pose 0.4565 (ultra .val на ONNX, rec_pose_full2) | 0.5841/0.4564 (full inmem) | = col2 (mini) ✅ |
  | seg | 0.5382/0.3615 | 0.5377/0.3596 (mini) | 0.5375/0.3598 (mini) ✅ |
  | cls | 0.700 | 0.70008 (full) | = col2 (same subset) ✅ |
  | obb | **0.5885/0.4444 (1024, val на std ONNX)** ✅ | **0.588/0.4447 (1024, full)** ✅ | **0.588/0.4447 (1024, full, == col2)** ✅ |
- **ВАЖНО про pose col1**: прежнее значение col1 pose = 0.608/0.465 — СТАЛОЕ (снято на БИТЫХ
  yolo-лейблах до их регенерации). Корректный col1 = rec_pose_full2 (0.5840/0.4565, ultra .val
  на том же std ONNX и тех же исправленных лейблах) — совпадает с col2 до 0.0001.
  **ULTRA_PT (ultra .val на .pt, rec_pose_pt.log, 2026-08-06, завершён)**: box_map50=0.5958,
  box_map50_95=0.4412, pose_map50=0.4552, pose_map50_95=0.2857.
- Осталось: RKNN деплой (int8), yolo26, yolo12.

### Этап D. yolo26
- Скачать yolo26n (.pt), сконвертировать std ONNX + RK-opt ONNX.
- Разобрать голову yolo26 (Detect26 etc.) и её постпроцесс (зонд выходов).
- Валидировать col1/col2/col3 по всем задачам. sem тоже.
- Учесть: `rknn_patch.py` уже содержит `_patch_detect26`-и-подобные для yolo26.

**Статус (2026-08-07)**:
- **detect ЗАВЕРШЁН**: col1 == col2 == col3 = **0.5589/0.4012** (full 5000; pycoco 0.56838/0.40854/ar100=0.60433, preds 747358).
- **pose ЗАВЕРШЁН**: col1 (ultra .val std) pose 0.4685/0.3236, box 0.5914/0.4178;
  col2 (eval std) inmem_box 0.5916/0.4196, inmem_pose 0.4685/0.3236; col3 == col2 байт-в-байт (149872 preds).
- **seg ЗАВЕРШЁН (mini 200)**: col1 (ultra .val std) mask 0.6061/0.3857; col2 inmem_seg 0.6062/0.3844;
  col3 == col2 байт-в-байт (30786 preds). ВАЖНО: для честного mini-сравнения нужен мини-json
  (load_coco идёт по image_id; --limit на полном json даёт другие imgs).
- **obb ЗАВЕРШЁН**: col1 0.586/0.447 == col2 == col3 0.5856/0.4464 (70607 preds, col3==col2 байт-в-байт).
  **Разгадка дельты 0.5803**: ультралайтс NMS time_limit=2.0+max_time_img*bs урезал preds на 2 кадрах.
  Честный col1 — с max_time_img=1000 (monkeypatch nms.non_max_suppression) → 0.586/0.447.
 - **cls ЗАВЕРШЁН**: col1 (ultra .val std ONNX) top1 **0.7139**; col2 = **0.71394**; col3 == col2 = **0.71394**
   (full 50000, errors 0, ~3200s каждый).
 - one2one-проблема решена: фикс `end2end=False` перед export в convert.py (fuse() иначе оставляет one2one-головы).
 - **sem (код ГОТОВ, 2026-08-07)**: eval.py semantic ветка реализована (compute_inmemory_sem_map — реплика
   SemanticMetrics; letterbox_semantic_gt pad=255; apply_cityscapes_label_mapping; run_3way_eval.py запись
   yolo26n-sem). **Эквивалентность подтверждена байт-в-байт**: ультралайтс .val() std ONNX mIoU 0.0108/0.0506
   == eval.py (smoke, вкл. per-class). semantic_postprocess исправлен (std карта не argmax-ится повторно;
   шаблон уже был исправлен — обновлены кэш и convert/postprocess.py).
   **Метод col1 подтверждён**: SemanticSegmentationValidator ЕСТЬ в 8.4.105 (models/yolo/semantic/val.py),
   .val() на std ONNX (task='semantic') работает, ключи 'metrics/mIoU', 'metrics/pixel_acc'.
    Осталось: получить Cityscapes (cityscapes8.yaml: masks_dir=masks, label_mapping labelIds→trainId, pad=255)
    и прогнать полный 3-way. **ВНИМАНИЕ для col3 (ИСПРАВЛЕНО 2026-08-07)**: rknn_patch._rknn_semantic_segment
    (rknn_patch.py:96) РАНЬШЕ заменял bilinear→nearest → col3 != col2 (0.5802/0.9549, 2.2% пикселей на тесте).
    **Фикс: переведён на bilinear align_corners=False (как std) + переэкспорт RK-opt ONNX** → col3 == col2
    байт-в-байт. RKNN-поддержка bilinear Upsample на NPU — проверить при RKNN-деплое (иначе упадёт на CPU).
  - **sem 3-way ЗАВЕРШЁН (2026-08-07, cityscapes8 4 imgs)**: col1 (ultra .val std ONNX) = **0.6104/0.9646**
    == col2 (eval std) == **col3 (eval RK-opt) 0.6104/0.9646** (все три байт-в-байт, вкл. per-class).
    **БАГ был в eval.py**: порог определения raw mask
    `gt.max() > 50` не срабатывал (cityscapes8 маски raw labelIds 7..33, max=33) → label_mapping не применялся →
    0.0565/0.0036. Фикс: порог `> 18`. **ВСЕ 6 ЗАДАЧ yolo26n ЗАКРЫТЫ** (detect/pose/seg/obb/cls/sem).
  - **sem 3-way ЗАВЕРШЁН — полный Cityscapes (500 val, 2026-08-08)**: **col1 (ultra .val std ONNX) = 0.7133/0.9458
    == col2 (eval std) == col3 (eval RK-opt) = 0.7133/0.9458** (per-class IoU col2 == col3 байт-в-байт,
    524288000 px, 0 errors, ~5 мин каждая колонка).
    **Источник данных — Antreas/Cityscapes (HF)**: parquet val split (3 шарда, `data/val-*`, ~1.2GB) — изображения
    в НОРМАЛЬНОМ PNG (в отличие от Chris1/cityscapes, где байты изображений в битом «SPIFF»-формате, нечитаемы
    ни PIL, ни cv2, ни PyAV, ни ffmpeg — 0/500 совпадений с официальными масками). Имена — официальные, берутся из
    path маски (`<city>_<seq>_<frame>_gtFine_labelIds.png` → stem совпадает с масками/изображениями в
    `datasets/cityscapes/images|masks/val/`). Проверено: маски из parquet байт-в-байт == gtFine_trainvaltest.zip.
    `setup_datasets.py` обновлён (Antreas, 3 шарда, официальные имена); битый `leftImg8bit.zip` удалён.
    **БАГ ООМ в eval.py (исправлен 2026-08-08)**: semantic-ветка копила ВСЕ 500 pred/gt-карт (1024² int64 ≈ 8MB ×
    1000 = ~8GB) → OOM-kill ~300 imgs. Фикс: инкрементальный confusion matrix (`np.add.at` per-image) +
    `sem_metrics_from_matrix()`; результат байт-в-байт тот же (проверено на 5 imgs 0.6179/0.9644 до и после).
  - Осталось: **yolo12** (detect/pose/seg/cls/obb/sem 3-way), RKNN-деплой, регрессия конвертера.

### Этап E. yolo12
- **yolo12n detect 3-way ЗАВЕРШЁН (2026-08-07, full 5000)**: col1 (ultra .val std ONNX) = **0.5567/0.3995**
  == col2 (eval std, inmem) == col3 (eval RK-opt) **0.5567/0.3995**; pycoco col2 == col3 = 0.56640/0.40720,
  ar100 0.60507, 775134 preds с обеих сторон, errors 0.
  YOLO12n: стандартная голова Detect (reg_max=16, DFL, end2end=False, выход [1,84,8400]) — RK-opt патчи и
  шаблоны yolo11 подошли БЕЗ изменений; отличие только в backbone (A2C2f attention).
  Официальных весов yolo12 для seg/pose/obb/cls НЕТ (только detect n/s/m/l/x) — yolo12 ограничен detect.

### Этап G. Sweep nano-моделей от версии 3 (2026-08-08)
Причина: пользователь хочет проверить эквивалентность не только yolov8n по всем задачам,
а ВСЕ непроверенные nano-модели, начиная с версии 3.

**Доступность nano-весов** (assets v8.4.0, `GITHUB_ASSETS_NAMES`):
- yolov3n / yolov6n / yolov9n — **НЕ существуют** (yolov9 имеет только yolov9t/s/m/c/e).
- **yolov5nu.pt** — есть (голова `Detect`, reg_max=16, no=144; decode через make_anchors как v8 → существующий пайплайн должен работать без изменений).
- **yolov8n** — есть (все задачи) + detect уже завершён (ниже).
- **yolov10n.pt** — есть (голова `v10Detect`, end2end=True; `export_std_onnx.py` форсирует end2end=False → raw [1,84,N]).
- yolo11n / yolo12n / yolo26n — уже ЗАВЕРШЕНЫ.

**yolov8n detect 3-way ЗАВЕРШЁН (2026-08-08, full 5000, coco_val2017)**:
- col1 (ultra .val std ONNX) = **map50 0.5158 / map50_95 0.3659**
- col2 (eval std, inmem) = **0.5158/0.3659** == col1 байт-в-байт
- col3 (eval RK-opt, inmem) = **0.5158/0.3659** == col2 байт-в-байт
- pycoco col2 == col3 = **0.52536/0.37324** (col2 881857 preds / 1474.7s, col3 881856 / 1436.6s —
  разница 1 pred = tie-break NMS; метрики равны до 1e-7)
- col1 на coco8 (4 imgs) для смоука: 0.808/0.6082.
- **Фикс col1**: `datasets/coco8` был СИМЛИНКОМ на хостовый абсолютный путь
  (`convert/datasets/coco8`) → внутри контейнера битый → `check_det_dataset` падал
  ("images not found, missing path ..."). Заменён реальной копией
  (`rm datasets/coco8 && cp -r convert/datasets/coco8 datasets/coco8`, 540K).
  УРОК: в `datasets/` НЕ должно быть симлинков на хостовые пути — только реальные директории.

**yolov8n seg/pose/cls/obb 3-way**: переведено на interleaved-режим (Этап H) — smoke-тесты всех
6 задач ПРОЙДЕНЫ (2026-08-08); mini/полные прогоны по желанию пользователя. Старый per-column
свип устарел (был на старом runner).

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
| yolov5nu | `Detect` | ✅ (в списке) |
| yolov8n (все задачи) | `Detect`/`Segment`/`Pose`/`OBB`/`Classify` | ✅ (свип) |
| yolov9t | `Detect` (reg_max=16) | ✅ патч Detect без изменений |
| yolov10n | `v10Detect` (end2end=True, наследует Detect НО переопределяет forward) | ⚠️ нужен отдельный `_patch_v10detect` + end2end ветка постпроцесса |
| yolo11n / yolo12n / yolo26n | Detect/26-головы | ✅ ЗАВЕРШЕНО |
| rtdetr-l/x | `RTDETRDecoder` (трансформер, NMS-free) | ❌ attention/softmax на NPU RK3588 — RK-opt патча нет; есть только задел (is_rtdetr → opset16 + detr_postprocess) |
| yoloe-* (v8/11/26) | `YOLOEDetect`/`YOLOESegment(26)` | ⚠️ open-vocabulary (промпты); ultralytics добавил prompt-free RKNN только в 8.4.113+ |
| yolov8-world | `WorldDetect` | ⚠️ open-vocabulary |
| yolo_nas_s/m/l | (super_gradients) | ❌ НЕ грузится: `super_gradients` не собрался в контейнере (битые wheels pycocotools/onnx); интеграция устаревшая (warning: будет удалена) |

**Итог**: критерий — **самая лёгкая модель из каждой линейки** (не обязательно nano; пользователь
уточнил 2026-08-08). В инвентаризацию 3-way добавляются **yolov3-tinyu** (самая лёгкая в yolo3) и
**yolov9t** (самая лёгкая в yolo9) — голова Detect, подойдут без изменений, как yolo12. **yolov10n** —
после доработки патча v10Detect.
RT-DETR / YOLOE / YOLO-World / YOLO-NAS — НЕ включаются (RKNN не способен либо open-vocabulary).

Зонды: `/tmp/opencode/probe_models.py|log`, `probe_heads.py|log`, `probe_v10.py|log`, `probe_v3.py|log`.

### Этап H. Interleaved runner (ТЕКУЩАЯ ЗАДАЧА, 2026-08-08)
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

**Статус: ВСЕ 6 smoke-тестов ПРОЙДЕНЫ (2026-08-08), col1==col2==col3, diverged=0**:
yolov8n detect 0.5465/0.4092 (limit 8), **mini 64 imgs 0.6101/0.4505, 10605 preds**;
yolov8n-seg segment 0.5492/0.3304 (1916 preds); yolov8n-pose pose 0.368/0.3031 (pose 0.331/0.2453);
yolo11n-obb obb 0.834/0.6207 (1024); yolov8n-cls classify top1 0.5; yolo26n-sem semantic mIoU
0.6168/pixacc 0.9583 (per-class col2==col3 байт-в-байт).

**Mini-64 прогоны ЗАВЕРШЕНЫ (2026-08-08, вечер)** — все 6 задач, ВЕЗДЕ col1==col2==col3,
errors 0, diverged 0 (логи `cache/_mini_{seg,pose,obb,cls,sem}.log`):
| задача | результат |
|---|---|
| detect | 0.6101/0.4505 (10605 preds, был ранее) |
| seg | 0.5579/0.3476 (11451 preds) |
| pose | box 0.4512/0.3156, pose 0.3286/0.2184 (2127 preds) |
| obb | 0.8315/0.6622 (8446 preds, 1024) |
| cls | top1 0.84375 |
| sem | mIoU 0.705/pixacc 0.9385 (per-class col2==col3 байт-в-байт) |

**Исправлено 4 бага (2026-08-08)**:
1. **nms-мутация std_raw**: `_nms_det` без `.copy()` портил буфер std_raw (xywh→xyxy инплейс через
   transpose-вью) → col2 ломался (0.0 метрик). Фикс `.copy()` (run_3way_eval.py:678).
2. **'masks' vs 'masks_160'**: canon_col1 писал 'masks', остальные ждали 'masks_160' → KeyError.
   Унифицировано на 'masks_160'.
3. **segment/pose постпроцессы клипали bbox ВСЕГДА** (не под `if not EVAL_PRECISE` как detect) →
   col1 vs col2 bbox diff до 1.24 px. Фикс в шаблонах + патч 19 сгенерированных кэш-файлов.
4. **semantic-ветка** использовала постпроцесс (orig-пространство) при GT в letterboxed → TypeError;
   переведена на raw argmax (`raw[0][0].argmax(0)`).

**ПОЛНЫЙ СВИП ВСЕХ ЛИНЕЕК ЗАВЕРШЁН (2026-08-09, ночь) — 9 прогонов, ВЕЗДЕ col1==col2==col3
(метрики байт-в-байт), errors 0**. Очередь `cache/_full_sweep.sh` (v8seg→v8pose→v8obb→v8cls→v5nu→
v3tiny→v9t) + `cache/_full_sweep2.sh` (v8det→v10n); master-лог `cache/_full_sweep_master.log`,
пер-таск логи `cache/_full_<tag>.log`:

| модель | задача | imgs | col1==col2==col3 | preds | elapsed | diverged 12/23 |
|---|---|---|---|---|---|---|
| yolov8n | detect | 5000 | 0.5158/0.3659 | 881857 | 3502s | 2/4 |
| yolov8n-seg | segment | 5000 | 0.486/0.2994 | 936734 | 7477s | 5/4 |
| yolov8n-pose | pose | 5000 | box 0.5948/0.4368, pose 0.4508/0.287 | 177575 | 3761s | 2/1 |
| yolov8n-obb | obb | 458 DOTA | 0.5828/0.4388 | 86509 | 920s | 0/0 (equal) |
| yolov8n-cls | classify | 50000 | top1 0.68936 | — | 6573s | 0/0 (equal) |
| yolov5nu | detect | 5000 | 0.4868/0.3355 | 937467 | 3768s | 2/2 |
| yolov3-tinyu | detect | 5000 | 0.4453/0.2857 | 890634 | 4842s | 2/0 |
| yolov9t | detect | 5000 | 0.5183/0.3727 | 862451 | 3702s | 3/3 |
| yolov10n | detect | 5000 | 0.5403/0.3869 | 779546 | 3249s | 1/1 |

**Diverged (всего 25 кадров из ~38k) — подтверждённый near-tie**: у ultralytics-NMS и нашего NMS при
почти/равном score (разница ≤ 1e-6) редкие кадры выбирают разных кандидатов (проба на 151000: два бокса
с score 0.001033 → col1 берёт [215.97,163.79,304.83,475.84], col2 — [218.23,148.13,312.15,429.39]);
кадр «40 vs 41» в повторе дал 40/40 (недетерминизм torch-NMS). Метрики при этом НЕ меняются (байт-в-байт).
**yolov10n подтверждён без патча**: голова v10Detect (end2end=True) экспортируется корректно —
`convert.py` ставит `m.end2end=False` всем модулям → fuse() не удаляет cv2/cv3; RK-opt совпал со std.

**Правило имён моделей**: std_model_dir/rko_model_dir строятся из МОДЕЛИ, а не task
(`yolov8n-seg segment`, `yolov8n-pose pose`, `yolov8n-cls classify`, `yolo11n-obb obb`,
`yolo26n-sem semantic`), иначе загружается detect-кэш → `std_raw[1]` IndexError.

**Далее**: (1) решить с пользователем — перегонять ли yolo11n/yolo12n/yolo26n полными interleaved
(раньше были полные 3-way на старом runner); (2) регрессия конвертера (Этап B) и RKNN-деплой;
(3) коммит накопленных изменений (git status: много modified + untracked).

### Этап F. Финальные артефакты
- Перегенерировать `cache/3way_results/*.json`.
- Обновить AGENTS.md (Work State) и PLAN.md.
- (sem уже реализован и подтверждён; осталось по желанию: полный Cityscapes 500 val, RKNN int8-деплой)

## 5. Тяжёлые факты (не потерять)

1. **val-NMS ultralytics — multi_label=True**: `models/yolo/detect/val.py postprocess`
   → `nms.non_max_suppression(..., multi_label=True, nc=self.nc, rotated=True)`.
   Каждый бокс × каждый класс с conf>thresh = отдельное предсказание. БЕЗ этого
   obb давал 75148 preds вместо 82057 (дельта map50 ~0.009).
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

## 6. Команды

```bash
# col2 (std ONNX)
docker run --rm --memory=4g --entrypoint python -w /app/percept \
  -v /home/orangepi/work/percept:/app/percept \
  -v /home/orangepi/work/percept/datasets:/app/datasets \
  percept-converter:latest \
  benchmark/eval.py detect datasets/coco_val2017 --model-dir cache/yolo11n_std --model-name yolo11n_std --backend cpu

# col1 (эталон) — /tmp/opencode/rec_*.py (rec_detect_full.py, rec_pose_full.py,
# rec_seg_mini.py, rec_obb.py), результаты в *.pkl

# Конвертация модели
./deploy.sh yolo26n-pose --fp16
# или прямо convert.py: docker compose run --rm convert yolo26n-pose --fp16
```

## 7. Ключевые файлы

- `benchmark/eval.py` — eval-движок: inmem box/pose/obb/seg, pycoco, GT per-axis.
- `pipeline.py`, `percept/inference/__init__.py` — пайплайн (часть проекта).
- `convert/convert.py` — конвертер; `_generate_postprocess` (шаблоны), `_deploy_to_model`.
- `convert/patches/rknn_patch.py` — патчи голов (RK-opt выходы), есть yolo26-патчи.
- `convert/templates/*_postprocess.py` — ШАБЛОНЫ (этап A: обновить до доказанной логики).
- `cache/yolo11n_{,_std,}{,-seg_std,-pose_std,-obb_std,-cls_std}/` — проверенные постпроцессы.
- `cache/3way_results/*.json` — финальные метрики.
- `/tmp/opencode/rec_*.py` — скрипты эталонов .val + pkl.
- `AGENTS.md` — Work State (автообновляется).
