FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# System dependencies required by OpenCV, video processing and Playwright
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libgl1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only PyTorch
RUN pip install --no-cache-dir \
    torch==2.12.1 \
    torchvision==0.27.1 \
    --index-url https://download.pytorch.org/whl/cpu

# Install Bowling AI dependencies
COPY requirements-docker.txt .

RUN pip install --no-cache-dir -r requirements-docker.txt

# Install Chromium for PDF report generation
RUN python -m playwright install --with-deps chromium

# Copy Bowling AI project
COPY . .

# Create API directories
RUN mkdir -p api_uploads api_results api_reports

EXPOSE 8000

CMD ["sh", "-c", "uvicorn api:app --host 0.0.0.0 --port ${PORT:-10000}"]
