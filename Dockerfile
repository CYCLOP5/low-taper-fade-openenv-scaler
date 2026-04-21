FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    bubblewrap \
    fuse-overlayfs \
    procps \
    iputils-ping \
    findutils \
    curl \
    ca-certificates \
 && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv==0.6.14

COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev --no-install-project

COPY README.md __init__.py client.py inference.py models.py hpc_gym.py openenv.yaml ./
COPY server ./server
COPY sysadmin_env ./sysadmin_env
COPY assets ./assets
COPY bench ./bench
COPY training ./training
COPY eval ./eval
COPY tools ./tools
COPY docs ./docs
COPY Makefile ./Makefile

RUN uv sync --locked --no-dev --no-editable

EXPOSE 8000

CMD ["uv", "run", "server", "--host", "0.0.0.0", "--port", "8000"]
