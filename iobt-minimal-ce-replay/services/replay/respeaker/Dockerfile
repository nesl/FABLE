FROM ubuntu:22.04

ARG DEBIAN_FRONTEND="noninteractive"
ENV TZ="America/Los_Angeles"

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip python3-dev make g++ libssl-dev libhdf5-dev libsndfile-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir --upgrade pip setuptools \
    && pip3 install --no-cache-dir pynng==0.8.1 paho-mqtt==2.1.0 h5py msgpack==1.1.0 \
       soundfile scipy numpy pandas

COPY services/replay/respeaker/app /app
COPY lib /lib/iobtmax
WORKDIR /app
CMD ["python3", "-u", "/app/app.py"]
