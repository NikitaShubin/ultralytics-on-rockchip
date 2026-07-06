FROM arm64v8/python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    wget unzip cmake build-essential \
    libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    ultralytics \
    rknn-toolkit2 \
    onnxruntime \
    onnxslim

RUN mkdir -p /app/coco8 && \
    wget -q https://github.com/ultralytics/assets/releases/download/v0.0.0/coco8.zip \
      -O /tmp/coco8.zip && unzip -q /tmp/coco8.zip -d /app/coco8 && rm /tmp/coco8.zip

COPY . /app/converter
WORKDIR /app/converter

ENTRYPOINT ["python", "/app/converter/converter.py"]
