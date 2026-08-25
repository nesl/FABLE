FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
WORKDIR /workspace/FABLE

COPY pyproject.toml README.md ./
COPY fable ./fable
COPY providers ./providers
COPY evaluation ./evaluation
RUN pip install --no-cache-dir .

COPY iobt-minimal-ce-replay/services/orchestration/fable_orchestrator/app /app
CMD ["python", "-u", "/app/app.py"]
