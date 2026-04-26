# ===========================================================================
# Dockerfile — Acoustic Anomaly Detection API (DCASE 2024 Task 2)
# ===========================================================================
# Multi-stage build: installs dependencies, copies source code and weights,
# then runs the FastAPI server on port 8000.
#
# BUILD:
#   docker build -t acoustic-anomaly-api .
#
# RUN (inference only — needs weights + processed data):
#   docker run -p 8000:8000 \
#     -v $(pwd)/weights:/app/weights \
#     -v $(pwd)/data:/app/data \
#     -v $(pwd)/configs:/app/configs \
#     acoustic-anomaly-api
#
# RUN (everything baked in — if you built with data included):
#   docker run -p 8000:8000 acoustic-anomaly-api
# ===========================================================================

FROM python:3.10-slim

# System deps for librosa (audio processing) and soundfile
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Copy source code
COPY api/ api/
COPY src/ src/
COPY configs/ configs/

# Copy weights and processed data if present (for baked-in images)
# These directories may be empty if you prefer volume mounts
COPY weights/ weights/
COPY data/processed_v1/ data/processed_v1/
COPY data/processed_v2/ data/processed_v2/

# Create required directories
RUN mkdir -p logs/predictions data/raw outputs

# Expose the API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=60s --timeout=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Run the FastAPI server
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
