FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    bubblewrap \
    proot \
    fuse-overlayfs \
    procps \
    iputils-ping \
    findutils \
    curl \
    ca-certificates \
 && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY __init__.py client.py inference.py models.py hpc_gym.py openenv.yaml ./
COPY server ./server
COPY sysadmin_env ./sysadmin_env
COPY assets ./assets
COPY bench ./bench
COPY training ./training
COPY eval ./eval
COPY tools ./tools
COPY docs ./docs
COPY Makefile ./Makefile

RUN python -m pip install --upgrade pip setuptools wheel \
 && python -m pip install .

EXPOSE 8000

CMD ["server", "--host", "0.0.0.0", "--port", "8000"]
