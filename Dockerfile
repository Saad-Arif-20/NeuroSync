FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
# In a production setup, we'd split requirements.txt to only install backend dependencies here.
RUN pip install --no-cache-dir fastapi uvicorn pydantic python-multipart numpy torch transformers

# Copy the backend code and models
COPY ./backend /app/backend
COPY ./models /app/models

# Expose port for FastAPI
EXPOSE 8000

# Start the Inference Server
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
