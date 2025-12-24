# Use Python 3.10 as base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install PyTorch CPU version first
RUN pip install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install remaining requirements from requirements.txt
# (pip will skip torch/torchaudio since they're already installed)
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Set Python to run in unbuffered mode for better logging
ENV PYTHONUNBUFFERED=1

# Default command - can be overridden
CMD ["python", "--version"]
