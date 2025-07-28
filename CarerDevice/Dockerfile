# Use Python 3.10 slim as base image
FROM python:3.10-slim

# Set noninteractive mode for apt-get
ARG DEBIAN_FRONTEND=noninteractive

# Install system dependencies required for building TenSEAL and Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake \
    g++ \
    git \
    curl \
    wget \
    build-essential \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy and install Python dependencies
COPY requirements.txt .

# Install TenSEAL and other dependencies in one step for better layer caching
RUN pip install --upgrade pip && \
    pip install --no-cache-dir git+https://github.com/OpenMined/TenSEAL.git && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of your app files
COPY . .

# (Optional) Create a non-root user and switch to it for better security
# RUN useradd --create-home appuser
# USER appuser

# Default command to run your app
CMD ["python", "app.py"]
