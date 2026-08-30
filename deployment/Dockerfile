FROM python:3.11-slim

WORKDIR /workspace/FABLE
COPY pyproject.toml README.md ./
COPY fable ./fable
COPY evaluation ./evaluation
COPY replay ./replay
COPY scripts ./scripts
RUN pip install --no-cache-dir .

ENTRYPOINT ["python"]
