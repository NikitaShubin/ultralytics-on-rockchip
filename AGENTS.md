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
**Эквивалентность eval.py vs ultralytics .val (col1/col2/col3)**: yolo26 — все 6 задач ЗАВЕРШЕНЫ (detect/pose/seg/cls/obb/sem), yolo12n detect ЗАВЕРШЁН, yolo11n — ЗАВЕРШЁН (все 5 задач, см. ниже). RKNN-деплой отложен по решению пользователя. Пайплайн и постпроцесс — часть проекта; постпроцесс обязан использоваться в конверторе (шаблоны convert/templates/). Полный план и статус — в `PLAN.md` (источник истины для следующего агента).

### Interleaved runner — ТЕКУЩАЯ ЗАДАЧА (2026-08-08)
- Цель: переписать `benchmark/run_3way_eval.py` на interleaved-режим (единственный): один цикл по кадрам,
  на каждом кадре col1 (ultralytics nms на том же std-тензоре), col2 (наш std постпроцесс), col3 (наш RK-opt
  постпроцесс); сразу сравнивать preds, копить только tp-агрегаты и счётчики расхождений; при расхождении
  сохранять только эти изображения (diverged-список).
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
- Статус: **ВСЕ 6 smoke-тестов ПРОЙДЕНЫ (2026-08-08), col1==col2==col3, diverged=0**:
  yolov8n detect 0.5465/0.4092 (limit 8), **mini 64 imgs 0.6101/0.4505, 10605 preds**; yolov8n-seg segment
  0.5492/0.3304 (1916 preds); yolov8n-pose pose 0.368/0.3031 (pose 0.331/0.2453); yolo11n-obb obb 0.834/0.6207
  (1024); yolov8n-cls classify top1 0.5; yolo26n-sem semantic mIoU 0.6168/pixacc 0.9583 (per-class col2==col3
  байт-в-байт). **Исправлено 4 бага** (см. Fixed 2026-08-08).
- **Mini-64 прогоны ЗАВЕРШЕНЫ для всех 6 задач (2026-08-08, вечер)**: detect 0.6101/0.4505 (10605 preds,
  был ранее), seg 0.5579/0.3476 (11451 preds), pose box 0.4512/0.3156 + pose 0.3286/0.2184 (2127 preds),
  obb 0.8315/0.6622 (8446 preds, 1024), cls top1 0.84375, sem mIoU 0.705/pixacc 0.9385 (per-class col2==col3
  байт-в-байт) — ВЕЗДЕ col1==col2==col3, errors 0, diverged 0. Логи: `cache/_mini_{seg,pose,obb,cls,sem}.log`.
  Полные прогоны (limit 500/5000) — по желанию пользователя.
- **ПОЛНЫЙ СВИП всех линеек ЗАВЕРШЁН (2026-08-09, ночь) — 9 прогонов, ВЕЗДЕ col1==col2==col3
  (метрики байт-в-байт), errors 0**. Очередь: `cache/_full_sweep.sh` (v8seg→v8pose→v8obb→v8cls→v5nu→v3tiny→v9t)
  + `cache/_full_sweep2.sh` (v8det→v10n). Master-лог `cache/_full_sweep_master.log`, пер-таск логи `cache/_full_<tag>.log`:
  - yolov8n detect (5000): 0.5158/0.3659, 881857 preds, 3502s, diverged 2/4
  - yolov8n-seg segment (5000): 0.486/0.2994, 936734 preds, 7477s, diverged 5/4
  - yolov8n-pose pose (5000): box 0.5948/0.4368, pose 0.4508/0.287, 177575 preds, 3761s, diverged 2/1
  - yolov8n-obb obb (458 DOTA, 1024): 0.5828/0.4388, 86509 preds, 920s, diverged 0/0, equal=true
  - yolov8n-cls classify (50000): top1 0.68936, 6573s, diverged 0/0, equal=true
  - yolov5nu detect (5000): 0.4868/0.3355, 937467 preds, 3768s, diverged 2/2
  - yolov3-tinyu detect (5000): 0.4453/0.2857, 890634 preds, 4842s, diverged 2/0, col2==col3 equal=true
  - yolov9t detect (5000): 0.5183/0.3727, 862451 preds, 3702s, diverged 3/3
  - yolov10n detect (5000): 0.5403/0.3869, 779546 preds, 3249s, diverged 1/1
  **Diverged (всего 25 кадров из ~38k) — подтверждённый near-tie**: у ultralytics-NMS и нашего NMS при
  почти/равном score (разница ≤ 1e-6) редкие кадры выбирают разных кандидатов (проба на 151000: два бокса с
  score 0.001033 → col1 берёт [215.97,163.79,304.83,475.84], col2 — [218.23,148.13,312.15,429.39]);
  кадр «40 vs 41» в повторе дал 40/40 (недетерминизм torch-NMS). Метрики при этом НЕ меняются (байт-в-байт).
  **yolov10n подтверждён без патча**: голова v10Detect (end2end=True) экспортируется корректно — `convert.py`
  ставит `m.end2end=False` всем модулям → fuse() не удаляет cv2/cv3; RK-opt совпал со std байт-в-байт.
- **ВАЖНО: имена моделей для задач** — std_model_dir/rko_model_dir строятся из МОДЕЛИ, а не task:
  `yolov8n-seg segment`, `yolov8n-pose pose`, `yolov8n-cls classify`, `yolo11n-obb obb`, `yolo26n-sem semantic`
  (иначе загружается detect-кэш `yolov8n_std` → `std_raw[1]` IndexError).
- Инфраструктура: **docker работает без sudo** (группа docker). Рабочий контейнер `segsmoke`
  (`percept-converter:latest`, torch 2.2.0, ultralytics 8.4.105), запуск:
  `docker run -d --name segsmoke --entrypoint sleep --memory=4g -v /home/orangepi/work/percept:/app -w /app percept-converter:latest infinity`
  (у образа ENTRYPOINT=convert.py — без `--entrypoint sleep` контейнер падает). Прогон: `docker exec -d segsmoke sh -c '... > /app/cache/_xxx.log 2>&1'`
  (bash-инструмент таймаутит 120s и убивает фоновые процессы; `-d` + лог в bind-папке cache/ — единственный паттерн).
  Контейнер `percept` (percept-percept) НЕ подходит (нет ultralytics).
- **НАЙДЕН КРИТИЧЕСКИЙ БАГ nms-мутации**: `non_max_suppression` делает `prediction[..., :4] = xywh2xyxy(...)`
  через `transpose`-вью инплейс; `torch.from_numpy(np.asarray(std_raw[0]))` БЕЗ копии → nms молча портит буфер
  std_raw (xywh→xyxy). `_nms_det` обязан использовать `.copy()` (run_3way_eval.py:678), иначе последующий
  `std_pp.postprocess(std_raw)` применяет xywh2xyxy повторно → сломанные bbox (col2=0.0 метрик, bbox diff 2.5e+02).

### Important Details
- **System freezes under sustained CPU load** (even 2 containers) due to 8GB RAM exhaustion + thermal throttling. Workaround: run 1 container at a time with `--memory=4g` or for eval use `--limit 500`.
- **YOLO26 export — end2end по умолчанию**: ультралайтс 8.4.105 экспортирует yolo26 в ONNX с NMS в графе (выход `[1,300,6]`), НЕ raw. `.val()` умеет оба формата: `DetectionPredictor.postprocess` → `non_max_suppression(..., end2end=model.end2end)`. Для честного сравнения (как yolo11) std ONNX экспортируется с `end2end=False` → raw (`benchmark/export_std_onnx.py:58`). End2end-проверка (col4) — опционально, по решению пользователя.
- **YOLO26 — НЕТ DFL**: box = 4 канала (ltrb-отступы) напрямую; `reg_max=1` → `NO_DFL=True` в шаблонах. RK-opt per-stride: тройка [box(4), cls(C, sigmoid), sm(1)] × nl; seg добавляет coeff(32)+proto(160); sem = один выход (см. ниже).
- **yolo26 RK-opt выходы**: detect [1,4]+[1,C]+[1,1] per stride; pose [1,4]+[1,1]+[1,51] (nc=1 person); seg [1,4]+[1,C]+[1,32] ×3 + proto[1,32,160,160]; obb [1,4]+[1,C]+[1,1] (angle в box, GRID [128,64,32], 1024); cls [1,1000] на 224; sem [1,19,1024,1024] (logits) для RK-opt, std sem = argmax-карта [1,1024,1024] (bake_argmax в SemanticSegment). semantic_postprocess обрабатывает ОБА формата (ndim==3 → argmax по каналу; ndim==2 → карта как есть).
- **RK-opt патчи = полная сеть** (airockchip-стиль): головы ОСТАЮТСЯ в графе (NPU), на CPU выносится только декодирование (DFL/dist2bbox/grid) + NMS. Это НЕ «отделение головы от бэкбона». Гипотеза «весь yolo26 на RKNN» (головы без DFL устойчивее к int8) — эмпирический вопрос, проверить позже (RKNN отложен).
- **Primary goal: eval.py on std ONNX must give EXACT same metrics as ultralytics .val() on the same model.** RK-opt is secondary priority.
- **RK-opt ONNX models have sigmoid baked into** `cls_scores` outputs (values are in 0-1 range). Postprocess must NOT apply `sigmoid()` to cls_scores. Only kpts confidence needs sigmoid.
- **RK-opt pose keypoint decoding must use YOLO11 standard formula**: `(offset * 2.0 + grid - 0.5) * stride`. The `if True:` branch (simple offset) is wrong.
- **Classify preprocessing**: ultralytics uses PIL Resize(224, bilinear) + CenterCrop(224) + ToTensor(/255). Pipeline's default letterbox is wrong for cls.
- **`/255.0` bug fixed**: `ModelBackend.infer()` no longer applies `/255.0` unconditionally. Instead, `/255.0` is applied only in the default letterbox preprocessing path. `custom_preprocess` returns [0,1] tensor directly.
- **val-NMS ultralytics — multi_label=True**: `models/yolo/detect/val.py postprocess` вызывает `nms.non_max_suppression(..., multi_label=True, nc=self.nc, rotated=True)`. Это даёт per-box-per-class расширение (каждый бокс × каждый класс с conf>thresh), т.е. у val БОЛЬШЕ предсказаний, чем у single-label постпроцесса. eval-постпроцессы detect/obb обязаны использовать multi-label expand (`np.where(cls > CONF_THRESH)` → x=[box[i], cls[i,j], j]) + class-aware NMS (offset `cls_id*MAX_WH`). Без этого obb давал 75148 preds vs эталон 82057 (delta map50 ~0.009).
- **obc NMS-семантика подтверждена**: с multi-label postprocess оба набора предсказаний байтово совпадают (P0003 145/145, всего на 8 imgs 1693/1693); полный прогон: 82057 preds с обеих сторон.
- **ЗАПРЕЩЕНО форсировать imgsz при экспорте/валидации** (2026-08-06): рабочее разрешение берётся из исходного .pt (args.imgsz) — конвертер так и делает (export без override), Pipeline детектит вход из модельного файла. Ad-hoc `export_obb640.py` (`model.args["imgsz"]=640`) переэкспортировал obb RK-opt вниз: 1024→640 даёт 0.477 вместо нативных ~0.61. Весь obb 3-way (col1/col2/col3) был снят на НЕнативном 640 → переделывается на 1024. Еval НЕ заменяет разрешение: eval.py:811 строит тот же `Pipeline`, input_size из ONNX shape (percept/inference/__init__.py:88-89) или INPUT_SIZE постпроцесса (:93-113).
- **Аудит входов ONNX (2026-08-06)**: все cache-модели на нативном разрешении КРОМЕ obb (640 vs 1024). detect/pose/seg=640 ✓, cls=224 ✓, прод model/model.onnx=224 ✓. Мусор `cache/yolo11n-cls.onnx` (640, не нативный) — удалён; `cache/yolo11n-obb.onnx` (640) — верхний уровень, тоже не нативный.
- **ВНИМАНИЕ (2026-08-07): `/tmp/opencode` ОЧИЩЕН при перезагрузке** — все бэкапы и ad-hoc скрипты утеряны (rkopt_1024/640.onnx, *_postprocess_640.py, rec_*.py, audit_onnx_sizes.py, mini_detect/, *.pkl, лог-файлы эталонов). Эталонные ЗНАЧЕНИЯ сохранены в этом файле и PLAN.md; RK-opt 1024 восстановлен в cache переэкспортом. Не полагаться на /tmp/opencode.

### Completed
- **yolo12n detect 3-way ЗАВЕРШЁН (2026-08-07, full 5000)**: col1 (ultra .val std ONNX) = **0.5567/0.3995**
  == col2 (eval std, inmem) == col3 (eval RK-opt) = **0.5567/0.3995**; pycoco col2 == col3 = 0.56640/0.40720,
  ar100 0.60507, 775134 preds с обеих сторон, errors 0. YOLO12n: стандартная голова Detect (reg_max=16, DFL,
  end2end=False, выход [1,84,8400]) — RK-opt патчи и шаблоны yolo11 подошли БЕЗ изменений; отличие только в
  backbone (A2C2f attention). Экспорт: std через export_std_onnx.py, RK-opt через convert --onnx (9 выходов
  per-stride [64,80,1]×3, NO_DFL=False). Официальных весов yolo12 seg/pose/obb/cls НЕТ — только detect n/s/m/l/x.
- **yolo26n detect 3-way ЗАВЕРШЁН (2026-08-07, full 5000)** — ВСЕ ТРИ КОЛОНКИ РАВНЫ:
  - col1 (ultra .val std ONNX) = col2 (eval std) = col3 (eval RK-opt) = **0.5589/0.4012**
  - pycoco: col2 == col3 = 0.56838/0.40854/ar100=0.60433; preds col2 747361, col3 747358
- **yolo26n obb 3-way ЗАВЕРШЁН (2026-08-07, 458 imgs)**: col1 (ultra .val std ONNX) = **0.586/0.447**
  == col2 == col3 = **0.5856/0.4464** (70607 preds, col3 == col2 байт-в-байт).
  **Разгадка прошлой дельты (0.5803 vs 0.586)**: ультралайтс NMS `time_limit=2.0+max_time_img*bs`
  (max_time_img=0.05) урезал preds на 2 кадрах (2 WARNING в obb_col1_onnx2.log) → заниженный col1.
  Повтор col1 с `max_time_img=1000` (полный NMS) → 0.586/0.447 ≈ col2. Доп. проверки: preds capture==eval
  (70607, max diff 6e-05), GT-маппинг per-axis==unit (0.5852), xyxyxyxy2xywhr_np == ultralytics ops (тот же minAreaRect),
  match_predictions (greedy) == наш _match_predictions_greedy. Ключевой вывод: для obb col1 УЛЬТРАЛАЙТС
  НЕ детерминирован под CPU-нагрузкой (NMS-таймаут) — сравнивать нужно при max_time_img высоком или через capture preds.
- **ПРОБЛЕМА one2one РЕШЕНА (2026-08-07)**: yolo26 end2end-модели: ультралайтс `model.fuse()` (Exporter, exporter.py:801-817) для end2end-голов УДАЛЯЕТ cv2/cv3 (one2many) и оставляет ТОЛЬКО one2one_cv2/one2one_cv3 → наш патч `_resolve_heads` неявно брал one2one → RK-opt давал другие логиты (col3 0.5627/0.3987, 618067 preds vs std). **Фикс**: в `export_onnx_only`/`export_and_compile` перед export ставим `m.end2end=False` на всех модулях + `model.end2end=False` → fuse() оставляет cv2/cv3, RK-opt = one2many как std. Переэкспортированы все 6 RK-opt ONNX (проверено: model.23 cv3=30, one2one=0).
- **yolo11n 3-way ЗАВЕРШЁН** (2026-08-06). Итог (RK-opt conversion loss: detect ~0.0002, остальные = 0):
  - detect: col1 0.542 / col2 0.532 (full 5000) / col3 0.5318 (full 5000, 614901 preds, inmem 0.5228/0.3678)
  - pose: col2 == col3 байт-в-байт (mini 200: pycoco 0.8519/0.6153/ar100 0.5882, inmem_box 0.6099/0.4553,
    inmem_pose 0.4893/0.296, 6112 preds); col2 full 0.7792/0.4803 pycoco, inmem_box 0.5841/0.4323
  - seg: col2 0.5377/0.3596 / col3 0.5375/0.3598 (mini 200); col1 mask 0.5382/0.3615
  - cls: col2 == col3 = 0.7598 (same 5000-subset первых 100 классов); col2 full 50000 = 0.70008; col1 0.700
  - obb: col2 == col3 = **0.588/0.4447 (1024, full 458 imgs, 78981 preds)** == col1 0.5885/0.4444
    (val на std ONNX). НЕнативный 640 давал заниженные 0.4768/0.3417 — переделано на 1024 ✅
- **Все col1 references collected**: detect 0.542, seg 0.484/0.462, obb 0.477, cls 0.700.
  Pose col1 = **0.584/0.4565** (rec_pose_full2, исправленные лейблы); старый 0.608/0.465 — битые лейблы, убрать.
  **ULTRA_PT (pose .pt, rec_pose_pt.log)**: box_map50=0.5958, box_map50_95=0.4412, pose_map50=0.4552, pose_map50_95=0.2857.
- **OBB эквивалентность ДОКАЗАНА** (evalobb_ml, dotav1 458 imgs): eval.py col2 **map50=0.4768, map50_95=0.3417, inmem 0.4769/0.3426**, preds **82057** (ровно как val-эталон 0.4769/0.3427, preds 82057). Фикс: multi_label=True NMS в obb-постпроцессе (single-label давал 75148 preds, inmem 0.4678/0.3367). До этого: GT-маппинг per-axis (sx=(isz-pad_w)/w, sy=(isz-pad_h)/h) — вклад +0.0004, NMS multi-label — основной.
- **obb_c1 SOLO re-run** gave map50=0.4765 ✅ (matching reference 0.4764)
- **POSE эквивалентность ДОКАЗАНА** (rec_pose_full2, 5000 imgs, regenerated labels): ultra inmem box 0.5840/0.4320, pose 0.4565/0.2852; pycocotools 0.7792/0.4807/ar100=0.4937 vs pose col2 0.5841/0.4323/0.4564/0.2851 и 0.7792/0.4803/0.4932. Дельта ~0.0001 — раундинг лейблов; прежняя дельта была из БИТЫХ yolo-лейблов.
- **DETECT эквивалентность ДОКАЗАНА** (mini_detect 200 imgs, 1460 anns, 80 classes, rebuilt из instances_val2017.json): eval inmem 0.5764/0.4236 (pycoco 0.5833/0.4294/ar100=0.6418) vs ultra RecDetectVal 0.5764/0.4237; 34783 preds с обеих сторон, greedy match 34782/34783 (img 11197 — NMS tie-break на равном score, разница ≤1.12px). Фикс: `EVAL_PRECISE=True` в `yolo11n_std_postprocess.py` (float bbox без int()/clip — ultralytics метрика не клипает).
- **SEG эквивалентность ДОКАЗАНА** (mini_seg 200 imgs): mask-метрика ultralytics из tp_m pkl = **0.5382/0.3615** (НЕ 0.5647/0.4298 — это был r.box!). eval.py c overlap-семантикой = **0.5377/0.3596**; pred-маски байтово идентичны (281/281 process_mask_160 == ops.process_mask). Дельта 0.0005/0.0019 — только float32-парсинг txt-лейблов ultralytics vs float64 json в eval.
- **mini_detect перестроен** (`build_mini_detect.py`): images/val2017 + labels/val2017 + annotations/instances_val2017.json + yaml 80 names.
- **`/255.0` bug fixed**: Removed `/255.0` from `ModelBackend.infer()` and moved it to default letterbox path only, so `custom_preprocess` returns final tensor.

### Active
- **Инвентаризация real-time моделей (2026-08-08)**: критерий — **самая лёгкая модель из каждой
  линейки** (не обязательно nano): добавлены **yolov3-tinyu** (23MB, самая лёгкая в yolo3; nl=2) и
  **yolov9t** (самая лёгкая в yolo9) — обе голова `Detect`, RK-opt патч подойдёт без изменений.
  **yolov10n** — после отдельного патча v10Detect.
  **НЕ включаются**: RT-DETR (трансформер, NPU RK3588 не тянет attention), YOLOE/YOLO-World (open-vocab),
  YOLO-NAS (super_gradients не собрался, легаси). Ultralytics НЕ обновлялась (8.4.105 → 8.4.116 новых
  моделей нет; эталоны сняты на 8.4.105). Детали — PLAN.md Этап G2.
- **yolo12n detect 3-way ЗАВЕРШЁН (2026-08-07, full 5000)**: col1 (ultra .val std) == col2 (eval std, inmem)
  == col3 (eval RK-opt) = **0.5567/0.3995**; pycoco col2 == col3 = 0.56640/0.40720 (775134 preds, errors 0).
  YOLO12n: стандартная голова Detect (reg_max=16, DFL, end2end=False) — RK-opt патчи yolo11 подошли без изменений.
  Официальных весов yolo12 для seg/pose/obb/cls НЕТ (только detect n/s/m/l/x) — yolo12 ограничен detect.
- **yolo26 все 6 задач ЗАВЕРШЕНЫ** (2026-08-07): detect 0.5589/0.4012, pose 0.4685/0.3236, seg 0.6061/0.3857,
  obb 0.586/0.447, cls 0.7139, sem 0.6104/0.9646 (cityscapes8) / **0.7133/0.9458 (полный Cityscapes 500 val,
  2026-08-08)** — все три колонки равны (см. Completed/Next Move).
- **sem-этап (реализован 2026-08-07, данные получены 2026-08-08)**: eval.py semantic ветка реализована
  (`compute_inmemory_sem_map` — реплика SemanticMetrics: confusion [gt,pred], pad GT=255, mIoU только по
  present классам, pixacc) + `letterbox_semantic_gt` (NEAREST к new_unpad=isz-pad + copyMakeBorder 255) +
  `load_cityscapes` + `apply_cityscapes_label_mapping`. **Подтверждено байт-в-байт на smoke-тесте**:
  ультралайтс `.val()` std ONNX (SemanticSegmentationValidator ЕСТЬ в 8.4.105, models/yolo/semantic/val.py)
  дал mIoU 0.0108/0.0506 == eval.py std 0.0108/0.0506 (вкл. per-class). semantic_postprocess исправлен:
  std [H,W] карта НЕ argmax-ится повторно (шаблон уже был исправлен, кэш/convert/postprocess.py починены).
  run_3way_eval.py: задача yolo26n-sem добавлена. **ВАЖНО для col3 (ИСПРАВЛЕНО 2026-08-07)**: `_rknn_semantic_segment`
  (convert/patches/rknn_patch.py:96) РАНЬШЕ заменял bilinear-апсэмпл на nearest → col3 != col2 (0.5802/0.9549).
  **Фикс: переведён на bilinear align_corners=False (как std)** + переэкспорт RK-opt ONNX → col3 == col2
  байт-в-байт (0.6104/0.9646, 0 px расхождений на 1M px smoke).
- **yolo26n-sem 3-way ЗАВЕРШЁН (2026-08-07, cityscapes8 4 imgs)**: col1 (ultra .val std ONNX) = **0.6104/0.9646**
  == col2 (eval std) == **col3 (eval RK-opt) = 0.6104/0.9646** (все три байт-в-байт, вкл. per-class).
  **БАГ был в eval.py**: порог определения raw
  mask был `gt.max() > 50`, а cityscapes8 маски — raw labelIds 7..33 (max 33) → label_mapping НЕ применялся →
  мусор 0.0565/0.0036. Фикс: порог `> 18` (trainId всегда 0..18). Данные: cityscapes8 скачан с github
  (ultralytics/assets), распакован в datasets/cityscapes8, yaml `datasets/cityscapes8-local.yaml` (с label_mapping).
  Для полного Cityscapes (500 val) нужен официальный датасет (регистрация) — не доступен по прямой ссылке.
- **yolo26n-sem 3-way ЗАВЕРШЁН — полный Cityscapes 500 val (2026-08-08)**: col1 (ultra .val std ONNX) =
  **0.7133/0.9458** == col2 (eval std) == col3 (eval RK-opt) = **0.7133/0.9458** (per-class IoU col2 == col3
  байт-в-байт, 524288000 px, 0 errors). **Источник данных — Antreas/Cityscapes (HF parquet val, 3 шарда,
  data/val-*, ~1.2GB)**: изображения в НОРМАЛЬНОМ PNG (Chris1/cityscapes — битый «SPIFF»-формат, нечитаем
  PIL/cv2/PyAV/ffmpeg). Имена официальные — из path маски (`<city>_<seq>_<frame>_gtFine_labelIds.png`), маски
  из parquet байт-в-байт == gtFine_trainvaltest.zip. setup_datasets.py обновлён; битый leftImg8bit.zip удалён.
  **БАГ ООМ в eval.py (исправлен)**: semantic-ветка копила все 500 pred/gt-карт (1024² int64 ~8GB) → OOM-kill
  ~300 imgs; фикс — инкрементальный confusion matrix per-image + `sem_metrics_from_matrix()` (тот же результат).
- **Метод col1 sem подтверждён**: `SemanticSegmentationValidator` ЕСТЬ в 8.4.105 (models/yolo/semantic/val.py),
  `.val()` на std ONNX (task='semantic') работает и даёт mIoU байт-в-байт == eval.py. Ключи результата:
  'metrics/mIoU' и 'metrics/pixel_acc'. Для полного 3-way нужен Cityscapes (маски trainId или raw + mapping).
- **run_3way_eval.py переписан под yolo26** (2026-08-07): TASKS → yolo26n-detect/seg/pose/obb/cls (пути на yolo26n), col1 = ultra .val на std ONNX для ВСЕХ задач (эталон; .pt-col1 не эталон), obb через task='obb', ветки метрик по cfg['eval_task']. resolve_imgsz из .pt; ROOT-пути.
- **yolo11n 3-way ЗАВЕРШЁН** — все 5 задач (detect/pose/seg/cls/obb) собраны, таблица в PLAN.md.
  Следующие шаги: RKNN-деплой (int8), регрессия конвертера (mini-метрики на конвертерных файлах), yolo26, yolo12.
- Оставшиеся бэкапы старых RK-opt постпроцессов: `/tmp/opencode/yolo11n_postprocess_old.py`,
  `/tmp/opencode/yolo11n_seg_postprocess_old.py`, `/tmp/opencode/yolo11n_pose_postprocess_old.py`,
  `/tmp/opencode/yolo11n_cls_postprocess_old.py`, `/tmp/opencode/yolo11n_obb_postprocess_old.py`,
  `/tmp/opencode/yolo11n-obb_rkopt_1024.onnx`.
- **Все 5 шаблонов синхронизированы и проверены** (detect/obb/pose/segment/classify):
  каждый обрабатывает ОБА формата (std single-output + RK-opt per-stride); `_generate_postprocess`
  генерирует все 5 без плейсхолдеров, py_compile OK; info из .pt корректна (nm=32, num_kpts=17,
  obb input 1024 → grid [128,64,32]).
- **pose col2 ПОЛНЫЙ** (eval.py std ONNX, 5000 imgs): pycocotools map50=0.7792, map50_95=0.4803, ar100=0.4932; **inmem_box 0.5841/0.4323, inmem_pose 0.4564/0.2851** (152185 pred, 0 errors, 1258с). ✅ Эквивалентно ultralytics-эталону (rec_pose_full2).
- **detect col2** (eval.py std ONNX, full 5000): **map50=0.532** (vs col1 0.542, delta ~1.8% = ONNX conversion loss; эквивалентность на mini подтверждена, pycoco 0.5833/0.4294 на mini).
- **seg col2/col3** — тяжелы для 8GB RAM (`--limit 200`); in-memory seg-метрика (overlap) + байтовые pred-маски подтверждены, полный прогон и col3 (RK-opt) впереди.

### Pose Investigation (2026-08-05) — РЕШЕНО ✅
**Корневая причина** дельты in-memory (0.4082/0.0674 vs эталон 0.6466/0.4794): **GT у ultralytics находится в letterboxed-пространстве (с паддингом), а НЕ в "mixed" (normalized×640 без паддинга)**.
- `Format(normalize=True)` нормализует bboxes/kpts по **letterboxed-изображению 640×640** (LetterBox уже применил scale+pad к инстансам).
- `_prepare_batch` затем умножает на imgsz (640,640) → GT возвращается в letterboxed 640-пространство — ТО ЖЕ пространство, что и предсказания. Никакого смешанного пространства нет!
- Моя реплика маппила GT как `orig_px/iw*640` (без pad) → систематический сдвиг на pad_top/pad_left → заниженный IoU/OKS.
- **Фикс**: в `benchmark/eval.py` `compute_inmemory_pose_map` GT-боксы и kpts теперь маппятся как `orig_px*scale + pad` (scale/pad из pipeline-метадаты, как у предсказаний) — буквально `gt_boxes = [bx*scale+pad_l, by*scale+pad_t, (bx+bw)*scale+pad_l, (by+bh)*scale+pad_t]`, kpts аналогично. x/iw, y/ih больше НЕ используется.
- Дополнительно подтверждено: letterbox percept `round`→`ceil` (ultralytics использует `min(math.ceil(w*scale), imgsz)`) → входы pixel-идентичны.
- **Верифицировано байт-в-байт** (mini_verify 200 imgs): `_match_predictions_greedy` = `match_predictions` (470+ предсказаний), tp/tp_p агрегация = `ap_per_class`, per-image counts 0 mismatch/200; eval.py inmem 0.6467/0.4730/0.4794/0.3084 vs эталон RecPoseVal 0.6466/0.4723/0.4794/0.3084.
- pycocotools на mini_verify (col2-стиль, только kp>0 персоны): map50=0.7935, map50_95=0.5042, ar100=0.4972, predictions=5685.

**col1 (0.608 box / 0.465 pose) vs col2 (0.779/0.480)** — это разные метрики и разные наборы персон: col1 = in-memory PoseMetrics на ВСЕХ персонах (включая kp=0) + OKS-matching; col2 = pycocotools COCOeval только на персонах с kp>0 + area-based box matching. Не сопоставимы напрямую. Сопоставимо: col1-стиль in-memory ONNX vs ultralytics-эталон на том же ONNX (0.5841/0.4564 — уточняется rec_pose_full).

### Segment/OBB in-memory реплики (2026-08-05, обновлено 2026-08-06)
- **КРИТИЧНО (mask_overlap=True по умолчанию)**: SegValidator val использует `Format(mask_overlap=hyp.overlap_mask=True)` → `polygons2masks_overlap((640,640), letterboxed_segments, ratio=4)` даёт **один (160,160) int-слой со значениями 1..N** (1 = инстанс с МАКСИМАЛЬНОЙ площадью, N = минимальной; маленькие рисуются поверх больших через `np.maximum`, i.e. "кто сверху"). Инстансы и cls сортируются по площади (desc). `_prepare_batch` декодирует per-instance: `masks = (overlap == i+1)` в порядке area-desc. Метрика: `mask_iou(gt_flat(n,25600), pred_flat(m,25600))` → (n,m) + `match_predictions` + `ap_per_class`.
- **Пример (img 134112)**: bed(59) площадью 17070px под dog(16)/laptop(63) → его декодированная маска = 8482px (минус перекрытые пиксели). Per-instance polygons2masks (как было раньше) даёт ДРУГУЮ метрику.
- **Реализовано в `benchmark/eval.py` `compute_inmemory_seg_map`**: `polygons2masks_overlap` + декод `(overlap==i+1)` + `gt_cls = gt_cls[order]` (order = argsort(-areas)). Pred: `masks_160` из постпроцесса (process_mask upsample=False). Fallback на polygons2masks при RLE.
- **Результат mini_seg 200 imgs**: eval inmem seg = **0.5377/0.3596**; маска-метрика ultralytics (из tp_m в seg_rec.pkl, ap_per_class по столбцам: col0=0.5382...) = **0.5382/0.3615**. Pred-маски байтово идентичны (281/281). Дельта 0.0005/0.0019 = float32-парсинг txt-лейблов (ultralytics) vs float64 json (eval). **ОСТОРОЖНО**: `r.box.map50` в rec_seg_mini — это BOX-метрика (0.5647/0.4298), НЕ mask!
- OBB: GT `rboxes = xyxyxyxy2xywhr` (orig) → letterboxed `bbox[..., :4].mul_(imgsz[[1,0,1,0]])` (cx,cy,w,h×640); метрика `batch_probiou` + `match_predictions` + `ap_per_class`. `ratio_pad` в pbatch = `((gain,gain),(pad_x,pad_y))`.
- **Реализовано в `benchmark/eval.py`**: `compute_inmemory_obb_map` (batch_probiou, GT/pred letterboxed [cx,cy,w,h,theta]) + `xyxyxyxy2xywhr_np` (numpy-реплика cv2.minAreaRect, правило w<h swap). Исправлен `gt_cls` (был все нули).
- **ОСТОРОЖНО**: в `cache/yolo11n-seg_std/yolo11n-seg_std_postprocess.py` main-mask кроп идёт по orig-bbox на 640-канвасе (после reverse letterbox) — не как ultralytics (кроп по letterboxed-bbox на 640). Для in-memory метрики это НЕ важно (используется masks_160), для pycocotools segm маска/кроп могут слегка отличаться от ultralytics-референса.

### Fixed (2026-08-08)
- **Interleaved runner — 4 бага исправлены (smoke-тесты 6 задач, см. «Interleaved runner — ТЕКУЩАЯ ЗАДАЧА»)**:
  (1) **nms-мутация std_raw**: `_nms_det` без `.copy()` портил буфер std_raw (xywh→xyxy инплейс) → col2 постпроцесс
  ломался (0.0 метрик). Фикс `.copy()` в run_3way_eval.py:678. (2) **'masks' vs 'masks_160'**: canon_col1 писал
  'masks', а canon_objs/_seg_tp_per_image/compare_preds ждали 'masks_160' → KeyError. Унифицировано на 'masks_160'.
  (3) **segment/pose постпроцессы клипали bbox ВСЕГДА** (не под `if not EVAL_PRECISE` как detect) → col1 vs col2
  bbox diff до 1.24 px. Фикс в шаблонах `convert/templates/{segment,pose}_postprocess.py` + патч 19 сгенерированных
  кэш-файлов (проверено: ALL OK — во всех кэш-постпроцессах clip под `if not EVAL_PRECISE`).
  (4) **semantic-ветка** использовала постпроцесс (возвращает orig-пространство) при GT в letterboxed →
  TypeError 'dict'; переведена на raw argmax (`raw[0][0].argmax(0)`) как eval.py (run_3way_eval.py:633-647).
- **yolov8n detect 3-way ЗАВЕРШЁН (2026-08-08, full 5000, coco_val2017)**: col1 (ultra .val std ONNX) = **0.5158/0.3659**
  == col2 (eval std, inmem) == **col3 (eval RK-opt, inmem) 0.5158/0.3659**; pycoco col2 == col3 =
  0.52536/0.37324 (col2 881857 preds, col3 881856 — разница 1 pred, tie-break; метрики равны до 1e-7).
  Финальная таблица (cache/3way_results/yolov8n-detect_col{1,2,3}.json): все три inmem байт-в-байт.
- **col1 падал с "Dataset ... images not found, missing path .../convert/datasets/coco8/images/val"**: `datasets/coco8` был **симлинком** на абсолютный хостовый путь `/home/orangepi/work/percept/convert/datasets/coco8` → внутри контейнера битый (target не существует) → `os.path.exists("/app/datasets/coco8")` = False → ultralytics `check_det_dataset` падал. **Фикс**: заменил симлинк на реальную копию (`rm datasets/coco8 && cp -r convert/datasets/coco8 datasets/coco8`, 540K). Проверено: `check_det_dataset("datasets/coco8-local.yaml")` → path `/app/datasets/coco8`. УРОК: в `datasets/` не должно быть симлинков на абсолютные хостовые пути — только реальные директории (внутри контейнера `/app/datasets/coco8` должен существовать как настоящая папка).

### Fixed (2026-07-30)
- **cls col2**: top1=0.70008 (50000 images) — matches col1 (0.700). Two bugs: (1) wrong preprocessing (PIL Resize(224)+CenterCrop(224)+ToTensor vs letterbox), (2) double `/255.0` in `infer()`.
- **detect eval mode**: now sets `MIN_WH=1, MAX_DET=300, DEDUP_ENABLED=False` to match ultralytics postprocessing. Previously `MIN_WH=40` filtered small objects.
- **postprocess files**: added `MAX_DET` and `DEDUP_ENABLED` to both `yolo11n_postprocess.py` and `postprocess.py`.
- **`load_yolo_as_coco` keypoints**: now extracts 17 keypoints (x,y,v) from YOLO pose labels (56-value format: `class cx cy w h x1 y1 v1 ... x17 y17 v17`). Previously ignored keypoints entirely — caused pose eval to return all -1.0.
- **`benchmark/eval.py` syntax**: восстановлена сигнатура `compute_inmemory_box_map` (случайно удалена правкой), `python -m py_compile` OK.

### Next Move
1. **Interleaved runner: smoke-тесты ПРОЙДЕНЫ (2026-08-08)** — все 6 задач: yolov8n detect (0.5465/0.4092,
   mini 64 imgs 0.6101/0.4505, 10605 preds), yolov8n-seg segment (0.5492/0.3304), yolov8n-pose pose
   (0.368/0.3031, pose 0.331/0.2453), yolo11n-obb obb (0.834/0.6207, 1024), yolov8n-cls classify (top1 0.5),
   yolo26n-sem semantic (0.6168/0.9583) — везде col1==col2==col3, diverged=0, equal=true.
   Дальше: mini-прогоны для остальных задач (segment/pose/obb/classify/semantic limit 64) и полные прогоны
   (limit 500/5000) по желанию пользователя; затем регрессия конвертера и RKNN-деплой (отложен).
2. **YOLO26 6-задач 3-way — ВСЕ ЗАКРЫТЫ (2026-08-07)**:
   detect: **col1 == col2 == col3 = 0.5589/0.4012 ✅**; pose: **col1 0.4685/0.3236 == col2 == col3 ✅**;
   seg: **col1 0.6061/0.3857 ≈ col2 == col3 ✅**; obb: **col1 0.586/0.447 ≈ col2 == col3 0.5856/0.4464 ✅**;
   cls: **col1 0.7139 == col2 == col3 0.71394 ✅** (50000 imgs);
   sem: **col1 == col2 == col3 = 0.6104/0.9646 ✅** (cityscapes8; все байт-в-байт; bilinear-фикс в rknn_patch).
   **sem полный Cityscapes 500 val (2026-08-08): col1 == col2 == col3 = 0.7133/0.9458 ✅** (per-class col2 == col3
   байт-в-байт, 0 errors; данные — Antreas/Cityscapes parquet; OOM-фикс инкрементального confusion в eval.py).
2. **YOLO12n detect 3-way ЗАВЕРШЁН (2026-08-07, full 5000)**: col1 (ultra .val std ONNX) = **0.5567/0.3995**
   == col2 (eval std, inmem) == **col3 (eval RK-opt) 0.5567/0.3995**; pycoco col2 == col3 = 0.56640/0.40720
   (775134 preds с обеих сторон, errors 0). YOLO12n = стандартная голова Detect (reg_max=16, DFL, end2end=False,
   выход [1,84,8400]), отличие от yolo11 только в backbone (A2C2f attention) — RK-opt патчи и шаблоны
   yolo11/detect подошли БЕЗ изменений. Официальных весов yolo12 для seg/pose/obb/cls НЕТ (только detect n/s/m/l/x),
   поэтому yolo12 ограничивается detect.
2. **obb на нативный 1024 — ЗАВЕРШЁН ✅** (2026-08-07, yolo11n): RK-opt 1024 (9 выходов per-stride) и std 1024,
   оба postprocess GRID [128,64,32]; **col1 0.5885/0.4444 (ultra .val на std ONNX, task='obb'),
   col2 0.588/0.4447, col3 0.588/0.4447 == col2 байт-в-байт (78981 preds)**. Мусор `cache/yolo11n-obb.onnx`
   (640) удалён. ВАЖНО: ультралайтс не определяет task для obb-ONNX — нужно `YOLO(onnx, task='obb')`,
   иначе val идёт как detect (нули). col1 на .pt даёт завышенные/другие метрики (систематика ~0.012) —
   эталон всегда ultra .val на std ONNX.
   **ВАЖНО для obb col1 (2026-08-07)**: ультралайтс NMS имеет time_limit (2.0+max_time_img*bs) и под
   CPU-нагрузкой урезает preds (2 кадра → дельта 0.005). Для честного col1 задавать max_time_img=1000
   (monkeypatch nms.non_max_suppression) или сравнивать по capture preds.

### Relevant Files
- `PLAN.md` — план/статус всей работы (источник истины)
- `convert/templates/detect_postprocess.py` — multi-label + class-aware NMS + EVAL_PRECISE, оба формата (std + per-stride)
- `convert/templates/obb_postprocess.py` — fast_nms (batch_probiou), multi-label, оба формата
- `convert/templates/pose_postprocess.py` — kpts `(offset*2.0+grid-0.5)*stride`, sigmoid(cls/kpts), оба формата
- `convert/templates/segment_postprocess.py` — process_mask_160 (без sigmoid, `>0`≡`sigmoid>0.5`), clip всегда, оба формата (std=2 выхода, RK-opt=10 с proto в конце); 854/854 и 607/607 байт-в-байт
- `convert/templates/classify_postprocess.py` — PIL Resize(224)+CenterCrop(224)+ToTensor, top-1 raw score
- `cache/yolo11n/postprocess.py` — **перегенерирован из detect-шаблона (2026-08-06)**; старый: `/tmp/opencode/yolo11n_postprocess_old.py`
- `cache/yolo11n-cls_std/{yolo11n-cls_std_,}postprocess.py`: PIL Resize(224)+CenterCrop(224)+ToTensor — matches ultralytics classify_transforms
- `cache/yolo11n-pose/postprocess.py`: fixed keypoint decoding (`if False:`) — removed double sigmoid from cls_scores
- `cache/yolo11n-seg_std/yolo11n-seg_std_postprocess.py` + `postprocess.py`: EVAL_PRECISE (float bbox, без clip), `process_mask_160` (ops.process_mask upsample=False), `masks_160` в результате
- `cache/yolo11n_std/yolo11n_std_postprocess.py` + `postprocess.py`: EVAL_PRECISE (float bbox, без clip) для detect; multi-label NMS (был с самого начала)
- `cache/yolo11n-obb/postprocess.py` — **перегенерирован из obb-шаблона (2026-08-06, GRID [80,40,20])** для RK-opt 640×640; будет переделан на 1024 (GRID [128,64,32]). Бэкапы 640: `/tmp/opencode/yolo11n-obb_postprocess_640.py`; старый (1024 single-label): `/tmp/opencode/yolo11n_obb_postprocess_old.py`
- `cache/yolo11n-obb_std/{yolo11n-obb_std_,}postprocess.py`: **multi-label NMS** (np.where(cls>CONF_THRESH) + class offset в batch_probiou) — исправлено 2026-08-06
- `benchmark/run_3way_eval.py` — **переписан на interleaved-режим (единственный, 2026-08-08)**: один цикл по кадрам (col1 nms / col2 std / col3 rko), `compare_preds` с допусками, инкрементальные tp-агрегаты, diverged-список; результат в `cache/3way_results/<model>-<task>_interleaved.json`; флаги `--limit/--force/--save-diverged/--diverged-dir`
- `benchmark/export_std_onnx.py` — штатный экспорт std ONNX на нативном imgsz (без override) + перегенерация postprocess
- `percept/inference/__init__.py`: `/255.0` moved from `infer()` to default letterbox path only
- `benchmark/eval.py`: MIN_WH=1/MAX_DET=300/DEDUP_ENABLED=False в eval mode; `compute_inmemory_box_map` (gt_cls исправлен), `compute_inmemory_pose_map` (GT=orig_px*scale+pad), `compute_inmemory_obb_map` + `xyxyxyxy2xywhr_np`, `compute_inmemory_seg_map` (polygons2masks_overlap + декод + gt_cls[order]); EVAL_DUMP_PREDS→`benchmark/dump_preds.pkl`; ветка detect/pose/segment собирает preds_by_img/scales_pads; **per-image tp-функции `_box/_pose/_seg/_obb_tp_per_image` + `_ap_finalize(prefix=...)`, `_seg_finalize` — обёртка над `_ap_finalize` (2026-08-08)**; `load_yolo_as_coco` возвращает 4 значения (`images, ann_by_img, cat_ids, tmp.name`)
- `/tmp/opencode/rec_pose_full.py` (лог rec_pose_full2.log), `/tmp/opencode/rec_detect_mini.py` (лог mini_ultra.log), `/tmp/opencode/rec_detect_full.py`, `/tmp/opencode/rec_seg_mini.py`, `/tmp/opencode/rec_obb.py`; `/tmp/opencode/mini_detect/`, `/tmp/opencode/mini_seg/` — 200 imgs + json + yaml; `/tmp/opencode/seg_rec.pkl`, `detect_rec.pkl`
- `/tmp/opencode/probe_batch.py` — зонд структуры pbatch: `ori_shape`=(h,w) готовый кортеж, `ratio_pad`=((gain,gain),(pad_x,pad_y)), `bboxes` в letterboxed 640-space
