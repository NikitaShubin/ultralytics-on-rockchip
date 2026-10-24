FROM python:3.11-slim

# pypi.org (Fastly SNI) в этой сети блокируется — pip ходит через зеркало.
# Переопределяется build-arg'ом, если зеркало недоступно.
ARG PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/
ENV PIP_INDEX_URL=${PIP_INDEX_URL} UV_INDEX_URL=${PIP_INDEX_URL}

RUN apt-get update && apt-get install -y --no-install-recommends \
    libv4l-0 ffmpeg && rm -rf /var/lib/apt/lists/*

RUN pip install uv
WORKDIR /app

COPY requirements.txt .
RUN uv pip install --system --no-cache -r requirements.txt
RUN uv pip install --system --no-cache rknn-toolkit-lite2

COPY percept/ percept/
COPY main.py app.py pipeline.py detector_frame.py config.py \
    camera_source.py video_loop.py tracker.py reid.py websocket_server.py ./
COPY templates/ templates/

CMD ["python3", "main.py"]
