FROM python:3.11-slim

ARG FABLE_MODEL_EXTRAS=0
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
WORKDIR /workspace/FABLE

RUN apt-get update \
    && apt-get install -y --no-install-recommends libxcb1 libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY fable ./fable
COPY providers ./providers
COPY iobt-minimal-ce-replay/lib /lib/iobtmax
COPY iobt-minimal-ce-replay/services/analytics/fable_multimodal_stack/app /app

RUN pip install --no-cache-dir '.[vehicle-tracking,multimodal-core]' \
        pynng==0.8.1 msgpack==1.1.0 && \
    if [ "$FABLE_MODEL_EXTRAS" = "1" ]; then pip install --no-cache-dir '.[multimodal-models]'; fi

CMD ["python", "-u", "/app/app.py"]
