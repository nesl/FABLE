# Requires NVIDIA Container Toolkit and an x86 CUDA-capable host.
FROM stereolabs/zed:5.4-runtime-cuda12.8-ubuntu22.04

ARG DEBIAN_FRONTEND="noninteractive"
ENV TZ="America/Los_Angeles"

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip python3-dev make g++ cmake libssl-dev libhdf5-dev \
    libglfw3 libglfw3-dev libcanberra-gtk-module ffmpeg x264 libx264-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir --upgrade pip setuptools \
    && pip3 install --no-cache-dir pynng==0.8.1 paho-mqtt==2.1.0 h5py msgpack==1.1.0 \
       pyzed==5.4 pandas opencv-python-headless numpy

COPY services/replay/zed/app /app
COPY lib /lib/iobtmax
COPY services/replay/common/replay_entrypoint.py /app/replay_entrypoint.py
WORKDIR /app
CMD ["python3", "-u", "/app/app.py"]
