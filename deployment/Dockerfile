ARG BASE_IMAGE=pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime
FROM ${BASE_IMAGE}

WORKDIR /workspace/FABLE
RUN apt-get update \
    && apt-get install -y --no-install-recommends iproute2 libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    YOLO_CONFIG_DIR=/tmp \
    PYTHONUNBUFFERED=1
COPY pyproject.toml README.md ./
COPY fable ./fable
COPY replay ./replay
COPY netwaggle ./netwaggle
COPY deployment ./deployment
COPY scripts ./scripts
RUN pip install --no-cache-dir '.[distributed,vision,audio]'
RUN python -c "import tensorflow_hub as hub; hub.load('https://tfhub.dev/google/yamnet/1')"
RUN python -c "from ultralytics import YOLO; YOLO('yolov8s.pt')"

ENTRYPOINT ["python"]
