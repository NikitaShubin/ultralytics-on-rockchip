"""Конвертер osnet_x1_0_market -> RKNN int8 (маршрут swdee/go-rknnlite).

Проверенный рабочий ReID-маршрут на RK3588 NPU: чекпоинт plain-market1501
(osnet_x1_0, 751 класс) деся Квай-чжоу (torchreid, MIT) из репо
swdee/go-rknnlite-build -> статический ONNX [1,3,256,128] opset 13 ->
RKNN int8 (mean/std = 255*(0.485/0.456/0.406)/(0.229/0.224/0.225),
дефолт-квант, калибровка 200 картинками Market1501 из того же репо,
rknn_batch_size=16). Качество: pairwise-corr с эталоном swdee batch8 = 1.0;
int8-NPU batch16 = 65-75 мс. Другая семья checkpoint-ов (msmt17_combineall)
на int8 ломается (collapse) — НЕ используем.

Артефакт кладётся в model/reid/osnet_x1_0_market.rknn (в гит не кладётся).
Запуск (из корня проекта):

  docker compose -f convert/docker-compose.yml run --rm \
      --entrypoint python3 convert /app/convert/reid_market.py

Зависимости: github.com доступен (HF не нужен); torch есть в образе convert.
"""
import os
import sys
import urllib.request
from pathlib import Path

ROOT = Path('/app/convert')
# Хостовый model/reid биндится в convert-контейнер как /app/convert/model/reid.
REID_DIR = ROOT / 'model' / 'reid'
WORK = Path(os.environ.get('REID_WORK_DIR', '/tmp/reid_market_work'))
PTH = 'osnet_x1_0_market_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip.pth'
PTH_URL = ('https://raw.githubusercontent.com/swdee/go-rknnlite-build/master/'
           'osnet-market1501/' + PTH)
OSNET_URL = ('https://raw.githubusercontent.com/KaiyangZhou/deep-person-reid/'
             'master/torchreid/models/osnet.py')
DATASET_URL = ('https://api.github.com/repos/swdee/go-rknnlite-build/'
               'git/trees/master?recursive=1')
RAW_URL = 'https://raw.githubusercontent.com/swdee/go-rknnlite-build/master/'


def _fetch(url: str, dst: Path, timeout: int = 600) -> None:
    if dst.is_file() and dst.stat().st_size > 0:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + '.part')
    req = urllib.request.Request(
        url, headers={'User-Agent': 'percept/1.0 (reid-market converter)'})
    with urllib.request.urlopen(req, timeout=timeout) as resp, open(tmp, 'wb') as f:
        while chunk := resp.read(1 << 20):
            f.write(chunk)
    tmp.replace(dst)
    print(f'sкачан: {dst.name} ({dst.stat().st_size} Б)')


def build_onnx() -> Path:
    import torch
    sys.path.insert(0, str(WORK))
    import torchreid_osnet as T

    pth = WORK / PTH
    onnx = WORK / 'osnet_x1_0_market_256x128.onnx'
    if onnx.is_file():
        return onnx
    model = T.osnet_x1_0(num_classes=751, pretrained=False)
    ckpt = torch.load(pth, map_location='cpu')
    sd = ckpt.get('state_dict', ckpt) if isinstance(ckpt, dict) else ckpt
    sd = {k.replace('module.', ''): v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        raise RuntimeError(f'чекпоинт не грузится: missing={missing[:3]}')
    if unexpected:
        print(f'unexpected keys (несущественно): {len(unexpected)}')
    model.eval()
    dummy = torch.randn(1, 3, 256, 128)
    torch.onnx.export(model, dummy, onnx, export_params=True,
                      opset_version=13, do_constant_folding=True,
                      input_names=['input'], output_names=['output'])
    print(f'ONNX собран: {onnx}')
    return onnx


def convert_rknn(onnx: Path) -> Path:
    from rknn.api import RKNN

    quant = os.environ.get('REID_QUANT', 'int8').lower()
    do_quant = (quant == 'int8')
    suffix = '' if do_quant else f'_{quant}'
    rknn_out = WORK / f'osnet_x1_0_market{suffix}.rknn'
    rk = RKNN(verbose=False)
    ret = rk.config(mean_values=[[255*0.485, 255*0.456, 255*0.406]],
                    std_values=[[255*0.229, 255*0.224, 255*0.225]],
                    target_platform='rk3588')
    assert ret == 0, ret
    ret = rk.load_onnx(model=str(onnx))
    assert ret == 0, ret
    batch = int(os.environ.get('REID_BATCH', '16'))
    ret = rk.build(do_quantization=do_quant,
                   dataset=str(WORK / 'subset.txt') if do_quant else None,
                   rknn_batch_size=batch)
    assert ret == 0, ret
    ret = rk.export_rknn(str(rknn_out))
    assert ret == 0, ret
    rk.release()
    print(f'RKNN собран: {rknn_out} (batch={batch}, {quant})')
    return rknn_out


def main() -> None:
    quant = os.environ.get('REID_QUANT', 'int8').lower()
    suffix = '' if quant == 'int8' else f'_{quant}'
    out = REID_DIR / f'osnet_x1_0_market{suffix}.rknn'
    REID_DIR.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    if out.is_file():
        print(f'уже на месте: {out}')
        onnx_done()
        return
    _fetch(OSNET_URL, WORK / 'torchreid_osnet.py')
    _fetch(PTH_URL, WORK / PTH)
    # калибровка: 200 картинок Market1501 из репо swdee
    ds_dir = WORK / 'dataset'
    ds_dir.mkdir(exist_ok=True)
    import json
    req = urllib.request.Request(DATASET_URL, headers={
        'User-Agent': 'percept/1.0 (reid-market converter)'})
    tree = json.loads(urllib.request.urlopen(req, timeout=600).read())
    names = [i['path'] for i in tree.get('tree', [])
             if i['path'].startswith('osnet-market1501/dataset/')
             and i['path'].endswith('.jpg')]
    if len(names) < 100:
        raise RuntimeError(f'в репо нет калибровочных картинок: {len(names)}')
    for p in names:
        _fetch(RAW_URL + p, ds_dir / p.split('/')[-1])
    with open(WORK / 'subset.txt', 'w') as f:
        for p in names:
            f.write(f'./dataset/{p.split("/")[-1]}\n')
    print(f'калибровка: {len(names)} картинок Market1501')

    onnx = build_onnx()
    rknn = convert_rknn(onnx)
    out.write_bytes(rknn.read_bytes())
    print(f'установлен: {out} ({out.stat().st_size} Б)')


def onnx_done() -> None:
    print('online weights уже есть, пропускаю сборку')


if __name__ == '__main__':
    main()
