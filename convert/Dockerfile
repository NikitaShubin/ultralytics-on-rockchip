FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    wget curl unzip libarchive-tools cmake build-essential \
    libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    ultralytics \
    rknn-toolkit2 \
    onnxruntime \
    onnxslim \
    pycocotools

COPY . /app/convert
WORKDIR /app/convert

# Нормализация расширений: rknn-toolkit2 поддерживает только .jpg .jpeg .png .bmp
RUN find /app/convert/datasets -name '*.JPEG' -exec sh -c 'mv "$1" "${1%.JPEG}.jpg"' _ {} \; 2>/dev/null || true
RUN find /app/convert/datasets -name '* copy*' -type f -delete 2>/dev/null || true

ENTRYPOINT ["python", "/app/convert/convert.py"]
