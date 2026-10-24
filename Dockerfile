FROM python:3.11-slim

WORKDIR /app

# System dependencies (ffmpeg for video, libgl/glib for OpenCV, build tools for insightface)
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libgl1 libglib2.0-0 build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first so this layer is cached between code changes
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY . .

# Run as a non-root user
RUN useradd --create-home appuser && chown -R appuser /app
USER appuser

ENV GRADIO_SERVER_NAME="0.0.0.0"
EXPOSE 7860
ENTRYPOINT ["python", "run.py"]
