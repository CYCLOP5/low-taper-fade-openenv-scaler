FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends bubblewrap fuse-overlayfs procps iputils-ping findutils curl ca-certificates && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . ./

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "sysadmin_env.server:app", "--host", "0.0.0.0", "--port", "8000"]
