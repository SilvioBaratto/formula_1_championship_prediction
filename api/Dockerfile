# API Dockerfile for F1 Championship Prediction FastAPI Backend
# Usage: docker build -t f1-api .
#        docker run -p 8000:8000 -v $(pwd)/../data:/app/data f1-api

FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements from root directory
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir fastapi uvicorn[standard] pydantic

# Copy project files
COPY src/ ./src/
COPY api/ ./api/
COPY scripts/ ./scripts/

# Create necessary directories
RUN mkdir -p data/models data/config data/outputs

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')"

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Run with uvicorn
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
