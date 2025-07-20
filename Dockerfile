# Mixture-of-Recursions (MoR) Research Environment
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the entire project
COPY . .

# Set Python path
ENV PYTHONPATH=/app

# Create non-root user for security
RUN useradd -m -u 1000 researcher && \
    chown -R researcher:researcher /app
USER researcher

# Default command
CMD ["python", "simple_mor_demo.py"]

# Labels for better organization
LABEL maintainer="your-email@example.com"
LABEL description="Mixture-of-Recursions transformer implementation"
LABEL version="1.0.0"
