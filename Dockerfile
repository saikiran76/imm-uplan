# ─────────────────────────────────────────────────────────────────────
# Uplan — HuggingFace Spaces Dockerfile (Gradio frontend only)
# ─────────────────────────────────────────────────────────────────────
# This runs on HuggingFace Spaces (free CPU tier).
# No GPU or ML libraries needed — all inference happens on the AMD server.
#
# Build:  docker build -t uplan-frontend .
# Run:    docker run -p 7860:7860 -e AMD_ENDPOINT=http://<amd-ip>:8000/extract uplan-frontend
# ─────────────────────────────────────────────────────────────────────

FROM python:3.11-slim

# Create non-root user (required by HuggingFace Spaces)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

# Install only frontend dependencies (lightweight — no torch!)
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY --chown=user app.py .
COPY --chown=user sample_outputs/ sample_outputs/

# Environment — HuggingFace Spaces will override AMD_ENDPOINT via Secrets
ENV USE_LIVE_BACKEND=true \
    AMD_ENDPOINT=http://localhost:8000/extract \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860

EXPOSE 7860

CMD ["python", "app.py"]
