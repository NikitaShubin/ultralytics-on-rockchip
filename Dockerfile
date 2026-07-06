FROM arm64v8/python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends libv4l-0 ffmpeg && rm -rf /var/lib/apt/lists/*

RUN pip install uv
WORKDIR /app

COPY requirements.txt .
RUN uv pip install --system --no-cache -r requirements.txt
RUN uv pip install --system --no-cache rknn-toolkit-lite2

COPY app.py pipeline.py .
COPY model/yolov8n_rkopt_o19.rknn /app/model/yolov8n_rkopt_o19.rknn

CMD ["python3", "app.py"]
