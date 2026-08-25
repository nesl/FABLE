# CPU-friendly YOLO container. It can run on GPU if the Docker runtime exposes CUDA,
# but the ZED replay container is the part that actually requires NVIDIA runtime.
FROM python:3.10-slim

ARG DEBIAN_FRONTEND="noninteractive"
ENV TZ="America/Los_Angeles" \
    OMP_NUM_THREADS=1 \
    YOLO_MODEL=/app/yolov8n.pt \
    YOLO_DEVICE=auto

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libhdf5-dev gcc g++ curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
    && pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu124 \
       torch==2.6.0 torchvision==0.21.0 \
    && pip install --no-cache-dir pynng==0.8.1 paho-mqtt==2.1.0 h5py msgpack==1.1.0 \
       numpy scipy opencv-python-headless pandas ultralytics==8.3.228

COPY services/analytics/yolo_detector/app /app
COPY lib /lib/iobtmax
WORKDIR /app
CMD ["python3", "-u", "/app/app.py"]
