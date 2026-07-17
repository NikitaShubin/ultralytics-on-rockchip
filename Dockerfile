FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libv4l-0 ffmpeg && rm -rf /var/lib/apt/lists/*

RUN pip install uv
WORKDIR /app

COPY requirements.txt .
RUN uv pip install --system --no-cache -r requirements.txt
RUN uv pip install --system --no-cache rknn-toolkit-lite2

COPY percept/ percept/
COPY main.py app.py pipeline.py utils.py detector_frame.py config.py \
    camera_source.py video_loop.py websocket_server.py ./
COPY templates/ templates/

CMD ["python3", "main.py"]
