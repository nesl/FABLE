FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
WORKDIR /workspace/FABLE

COPY pyproject.toml README.md ./
COPY fable ./fable
COPY providers ./providers
RUN pip install --no-cache-dir '.[vehicle-tracking]'

CMD ["python", "-u", "-m", "providers.vehicle"]
