FROM python:3.10-slim

ARG DEBIAN_FRONTEND="noninteractive"
ENV TZ="America/Los_Angeles"

RUN pip install --no-cache-dir --upgrade pip setuptools \
    && pip install --no-cache-dir paho-mqtt==2.1.0

COPY services/orchestration/complex_event_detector/app /app
WORKDIR /app
CMD ["python3", "-u", "/app/app.py"]
