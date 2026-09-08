FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
# Hugging Face Spaces runs on port 7860
ENV PORT=7860

WORKDIR /app

# Install system dependencies required by OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project source (weights are excluded via .gitignore and downloaded at runtime)
COPY . .

EXPOSE $PORT

# Hugging Face Spaces passes PORT as an env variable
CMD uvicorn api.app:app --host 0.0.0.0 --port $PORT
