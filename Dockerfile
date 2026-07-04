FROM arm64v8/python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends libv4l-0 ffmpeg && rm -rf /var/lib/apt/lists/*

RUN pip install uv
WORKDIR /app

COPY requirements.txt .
RUN uv pip install --system --no-cache -r requirements.txt

COPY app.py pipeline.py .

CMD ["python3", "app.py"]
