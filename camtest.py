import glob
import os
import select
import subprocess
import time

DEV_NONE = (os.devnull, 'w')


def try_ffmpeg(dev, label, cmd):
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        r, _, _ = select.select([proc.stdout], [], [], 1.0)
        ok = bool(r)
        data = b''
        err = b''
        if ok:
            data = proc.stdout.read(65536)
        proc.kill()
        proc.wait(timeout=2)
        err = proc.stderr.read().decode(errors='replace')
        return ok, data, err
    except Exception as e:
        return False, str(e).encode(), b''


def check_device(label, dev):
    print(f'\n=== {label}: {dev} ===')
    print(f'  os.path.exists ........... {os.path.exists(dev)}')
    print(f'  os.access(F_OK) .......... {os.access(dev, os.F_OK)}')
    try:
        fd = os.open(dev, os.O_RDONLY | os.O_NONBLOCK)
        os.close(fd)
        print('  os.open .................. OK')
    except FileNotFoundError:
        print('  os.open .................. FileNotFoundError')
    except OSError as e:
        print(f'  os.open .................. OSError({e.errno}) {e.strerror}')
    print(f'  in glob .................. {dev in glob.glob("/dev/video[0-9]*")}')
    print(f'  glob all ................. {sorted(glob.glob("/dev/video[0-9]*"))}')

    ok, data, err = try_ffmpeg(
        dev,
        'mjpeg+copy',
        [
            'ffmpeg',
            '-fflags',
            'nobuffer',
            '-flags',
            'low_delay',
            '-analyzeduration',
            '0',
            '-probesize',
            '32',
            '-f',
            'v4l2',
            '-input_format',
            'mjpeg',
            '-video_size',
            '640x480',
            '-framerate',
            '15',
            '-i',
            dev,
            '-f',
            'image2pipe',
            '-vcodec',
            'copy',
            '-',
        ],
    )
    print(
        f'  ffmpeg mjpeg+copy ........ {"OK" if ok else "TIMEOUT/FAIL"} (got {len(data)}B)',
    )
    if data:
        print(f'    data[:32] = {data[:32].hex()}')
    if err:
        print(f'    stderr = {err}')

    ok, data, err = try_ffmpeg(
        dev,
        'soft+encode',
        [
            'ffmpeg',
            '-fflags',
            'nobuffer',
            '-flags',
            'low_delay',
            '-analyzeduration',
            '0',
            '-probesize',
            '32',
            '-f',
            'v4l2',
            '-video_size',
            '640x480',
            '-framerate',
            '15',
            '-i',
            dev,
            '-f',
            'image2pipe',
            '-c:v',
            'mjpeg',
            '-q:v',
            '5',
            '-',
        ],
    )
    print(
        f'  ffmpeg soft+encode ........ {"OK" if ok else "TIMEOUT/FAIL"} (got {len(data)}B)',
    )
    if data:
        print(f'    data[:32] = {data[:32].hex()}')
    if err:
        print(f'    stderr = {err}')


print('Camera detection test')
print('Plugin/unplug camera on my command')
print()

while True:
    devices = sorted(glob.glob('/dev/video[0-9]*'))
    print(f'[{time.strftime("%H:%M:%S")}] devices={devices}')

    for i, dev in enumerate(devices):
        check_device(f'dev{i}', dev)

    print()
    time.sleep(2)
