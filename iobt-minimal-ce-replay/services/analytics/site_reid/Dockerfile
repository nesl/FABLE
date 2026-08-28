FROM python:3.11-slim

ARG FASTREID_COMMIT=c9bc3ceb2f7a6438b62fb515ea3df6d1e999e95d
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/opt/fastreid:/workspace/FABLE
WORKDIR /workspace/FABLE

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 gcc g++ curl \
    && rm -rf /var/lib/apt/lists/*
RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel \
    && pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cu124 \
      torch==2.6.0+cu124 torchvision==0.21.0+cu124 \
    && pip install --no-cache-dir paho-mqtt==2.1.0 numpy opencv-python-headless \
      scikit-learn==1.6.1 yacs termcolor tabulate
RUN pip install --no-cache-dir faiss-cpu==1.10.0
# The pinned upstream revision predates Python 3.10's collections move.
RUN mkdir -p /opt/fastreid \
    && curl -fL --retry 3 \
       "https://github.com/JDAI-CV/fast-reid/archive/${FASTREID_COMMIT}.tar.gz" \
       | tar -xz --strip-components=1 -C /opt/fastreid \
    && test -f /opt/fastreid/fastreid/modeling/meta_arch/baseline.py \
    && sed -i 's/from collections import Mapping, OrderedDict/from collections import OrderedDict\nfrom collections.abc import Mapping/' \
       /opt/fastreid/fastreid/evaluation/testing.py \
    && sed -i 's/from collections import Mapping/from collections.abc import Mapping/' \
       /opt/fastreid/fastreid/data/build.py

COPY pyproject.toml README.md ./
COPY fable ./fable
COPY providers ./providers
COPY iobt-minimal-ce-replay/config/reid /app/reid
RUN pip install --no-cache-dir -e .

CMD ["python", "-u", "-m", "providers.vehicle.descriptor_service"]
