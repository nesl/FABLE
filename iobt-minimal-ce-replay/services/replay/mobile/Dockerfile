FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir opencv-python-headless numpy paho-mqtt==2.1.0 pynng==0.8.1 msgpack==1.1.0

COPY lib /lib/iobtmax
COPY services/replay/mobile/app /app
WORKDIR /app
CMD ["python", "-u", "/app/app.py"]
