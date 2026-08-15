# Use a lightweight Python image.
FROM python:3.11-slim

# Set the working directory inside the container.
WORKDIR /app

# Copy the requirements file first.
# This helps Docker cache dependencies between builds.
COPY requirements.txt .

# Install Python dependencies.
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code.
COPY src ./src
COPY api ./api

# Copy the trained model and metadata.
COPY models ./models

# Make src available for imports.
ENV PYTHONPATH=/app/src

# Expose the FastAPI port.
EXPOSE 8000

# Start the FastAPI application.
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]