FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
WORKDIR /workspace/FABLE

RUN apt-get update \
    && apt-get install -y --no-install-recommends libxcb1 libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY fable ./fable
COPY providers ./providers
RUN pip install --no-cache-dir '.[vehicle-tracking]'

CMD ["python", "-u", "-m", "providers.vehicle"]
