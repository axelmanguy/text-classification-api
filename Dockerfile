# Use a lightweight Python base image
FROM python:3.10-slim

# Set environment variables to avoid writing bytecode and force UTF-8 encoding
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Create a non-root user and group
RUN groupadd --system appgroup && useradd --system --group appgroup appuser

# Set the working directory
WORKDIR /app

# Copy only necessary files first (for caching efficiency)
COPY requirements.txt .
COPY src/ src/

# Install dependencies as root, then clean up
RUN pip install --no-cache-dir -r requirements.txt

# Change ownership of the app directory to the non-root user
RUN chown -R appuser:appgroup /app

# Switch to non-root user
USER appuser

# Expose FastAPI's default port
EXPOSE 8000

# Command to start the API uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
