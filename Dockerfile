FROM python:3.10-slim

# Set environment variables to optimize Python & memory
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MALLOC_ARENA_MAX=2 \
    TF_USE_LEGACY_KERAS=1

WORKDIR /app

# Install system dependencies needed for compiling certain python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies first (leverage Docker cache)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    sed -i '/-e \./d' requirements.txt && \
    pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Install the project as a package
RUN pip install .

# Expose API port
EXPOSE 8000

# Run uvicorn with optimized workers & threads
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--limit-concurrency", "50"]
