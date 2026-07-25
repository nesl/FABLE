FROM python:3.11-slim

ARG FABLE_MODEL_EXTRAS=0
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
WORKDIR /workspace/FABLE

COPY pyproject.toml README.md ./
COPY fable ./fable
COPY providers ./providers
COPY iobt-minimal-ce-replay/lib /lib/iobtmax
COPY iobt-minimal-ce-replay/services/analytics/fable_multimodal_stack/app /app

RUN pip install --no-cache-dir '.[vehicle-tracking,multimodal-core]' && \
    if [ "$FABLE_MODEL_EXTRAS" = "1" ]; then pip install --no-cache-dir '.[multimodal-models]'; fi

CMD ["python", "-u", "/app/app.py"]
