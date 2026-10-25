# CPU image with two targets:
#   docker build -f docker/Dockerfile --target demo  -t wakeword-demo .
#   docker build -f docker/Dockerfile --target train -t wakeword-train .
FROM python:3.11-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 MPLBACKEND=Agg
RUN apt-get update && apt-get install -y --no-install-recommends libsndfile1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
RUN pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY wakeword/ wakeword/
COPY scripts/ scripts/
COPY apps/ apps/
COPY configs/ configs/
COPY tests/ tests/

FROM base AS train
COPY requirements-mlops.txt requirements-dev.txt ./
RUN pip install -r requirements-mlops.txt pytest
CMD ["bash"]

FROM base AS demo
EXPOSE 7860 7861
# mount trained checkpoints at /app/runs (docker-compose does this)
CMD ["python", "apps/demo_app.py", "--runs", "runs", "--host", "0.0.0.0", "--port", "7860"]
