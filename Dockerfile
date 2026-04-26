# ===========================================================================
# Dockerfile — Acoustic Anomaly Detection API (DCASE 2024 Task 2)
# ===========================================================================
# Lean build: installs only runtime dependencies, copies source code,
# model weights, configs, and the web UI. No raw/processed data included.
#
# Expected image size: ~1.5 GB (Python base + PyTorch CPU + weights)
#
# BUILD:
#   docker build -t acoustic-anomaly-api .
#
# RUN:
#   docker run -p 8000:8000 acoustic-anomaly-api
#
# Then open http://localhost:8000 in your browser.
# ===========================================================================

FROM python:3.10-slim

# System deps for librosa (audio processing) and soundfile
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (layer caching)
# Install CPU-only PyTorch FIRST to avoid the 2.5 GB GPU version
COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir --no-deps -r requirements.txt && \
    pip install --no-cache-dir -r requirements.txt

# Copy source code and web UI
COPY api/ api/
COPY src/ src/
COPY configs/ configs/

# Copy model weights (~75 MB)
COPY weights/ weights/

# Copy documentation
COPY README.md .

# Create required directories
RUN mkdir -p logs/predictions data/raw outputs

# Expose the API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=60s --timeout=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Run the FastAPI server
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
