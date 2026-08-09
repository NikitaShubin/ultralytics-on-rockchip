#!/usr/bin/env python3
"""Экспорт std ONNX (обычный ultralytics export, БЕЗ RK-opt патчей) в cache/{name}_std/.

Сохраняет НАТИВНОЕ разрешение модели (args.imgsz из .pt) — без override imgsz,
иначе метрики искусственно занижаются (см. obb: 1024 vs 640 → 0.61 vs 0.477).
Затем перегенерирует postprocess.py под фактический input_size.

Использование: python benchmark/export_std_onnx.py <model_name> [--pt PATH]
"""

import os
import sys
import json
import shutil
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ['ULTRALYTICS_NO_UPDATE_CHECK'] = 'True'


def export_std_model(name: str, pt: Path, dst_dir: Path) -> tuple[int, dict]:
    """Экспорт std ONNX из .pt в dst_dir + генерация postprocess.py.

    Сохраняет НАТИВНОЕ разрешение модели (args.imgsz из .pt) — без override imgsz.
    Возвращает (imgsz, info) — info как из _detect_model_info.
    """
    from ultralytics import YOLO
    model = YOLO(str(pt))
    m_args = getattr(model.model, 'args', {})
    imgsz = m_args.get('imgsz', 640) if hasattr(m_args, 'get') else getattr(m_args, 'imgsz', 640)
    if isinstance(imgsz, (list, tuple)):
        imgsz = imgsz[0] if imgsz else 640
    if not isinstance(imgsz, int) or imgsz <= 0:
        imgsz = 640
    print(f'[{name}] нативный imgsz={imgsz} (из .pt args)', flush=True)

    # Экспорт в отдельный каталог, чтобы НЕ затереть RK-opt ONNX в cache/{name}/.
    # ВАЖНО: ultralytics (8.4.x) сохраняет .onnx РЯДОМ С .pt (Exporter.file = model.pt_path),
    # игнорируя project/name kwargs. Поэтому экспортируем КОПИЮ .pt из tmp-каталога.
    tmp = ROOT / 'cache' / '_tmp_std_export'
    tmp.mkdir(exist_ok=True)
    pt_copy = tmp / pt.name
    shutil.copy2(str(pt), str(pt_copy))
    model_tmp = YOLO(str(pt_copy))
    # end2end=False: yolo26/v10 по умолчанию экспортируется с NMS внутри (выход [1,300,6]).
    # Для честного сравнения (как yolo11) нужен raw-выход: [1, 4+nc, total], который
    # декодирует наш постпроцесс, а ultralytics навешивает свой NMS при .val().
    model_tmp.export(format='onnx', opset=12, end2end=False)
    src = pt_copy.with_suffix('.onnx')
    if not src.exists():
        raise RuntimeError(f'экспорт не дал файла: {src}')

    dst_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(src), str(dst_dir / 'model.onnx'))
    shutil.copy2(str(src), str(dst_dir / f'{name}_std.onnx'))
    (dst_dir / 'model_name.txt').write_text(name + '\n')

    # Постпроцесс под фактический input_size
    from convert.convert import _detect_model_info, _generate_postprocess
    info = _detect_model_info(str(pt))
    if info['input_size'] != imgsz:
        raise RuntimeError(f'imgsz mismatch: .pt args={imgsz}, info={info["input_size"]}')
    pp = _generate_postprocess(info, dst_dir)
    shutil.copy2(pp, str(dst_dir / f'{name}_std_postprocess.py'))

    shutil.rmtree(tmp, ignore_errors=True)

    print(json.dumps({
        'status': 'OK', 'model': name,
        'input_size': imgsz, 'grid_sizes': info['grid_sizes'],
        'onnx': str(dst_dir / 'model.onnx'),
        'postprocess': str(dst_dir / 'postprocess.py'),
    }))
    return imgsz, info


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('model_name')
    ap.add_argument('--pt', default='', help='путь к .pt (по умолчанию cache/{name}/{name}.pt)')
    args = ap.parse_args()
    name = args.model_name

    if args.pt:
        pt = Path(args.pt)
    else:
        pt = ROOT / 'cache' / name / f'{name}.pt'
        if not pt.exists():
            pt = ROOT / 'cache' / f'{name}.pt'
    if not pt.exists():
        print(json.dumps({'status': 'FAIL', 'error': f'.pt не найден: {pt}'}))
        sys.exit(1)

    try:
        export_std_model(name, pt, ROOT / 'cache' / f'{name}_std')
    except Exception as e:
        print(json.dumps({'status': 'FAIL', 'error': str(e)}))
        sys.exit(1)


if __name__ == '__main__':
    main()
