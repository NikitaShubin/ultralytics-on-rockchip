FROM arm64v8/python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    wget unzip libarchive-tools cmake build-essential \
    libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    ultralytics \
    rknn-toolkit2 \
    onnxruntime \
    onnxslim \
    pycocotools

COPY . /app/converter
WORKDIR /app/converter

ENTRYPOINT ["python", "/app/converter/converter.py"]
