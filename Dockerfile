FROM python:3.11-slim

# HuggingFace Spaces requires uid 1000
RUN useradd -m -u 1000 user
WORKDIR /app

# Install dependencies first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# Copy source
COPY --chown=user:user . .

USER user

# Port 7860 is required by HuggingFace Spaces
EXPOSE 7860

ENV GRADIO_SERVER_PORT=7860
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Healthcheck — validator calls POST /reset
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD curl -sf -X POST http://localhost:7860/reset \
        -H "Content-Type: application/json" -d '{}' || exit 1

# Start FastAPI + Gradio via uvicorn
CMD ["python", "app.py"]
