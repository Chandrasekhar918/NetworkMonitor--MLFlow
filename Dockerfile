# Use an official Python runtime as a parent image
FROM python:3.11

# Set the working directory inside the container
WORKDIR /app

# Copy all files (including model files) into /app
COPY . /app

# Verify model files exist after COPY
RUN ls -l /app  # Debugging step

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir mlflow fastapi uvicorn numpy pydantic scikit-learn


# Expose port 8080 (Required for Google Cloud Run)
EXPOSE 8080  # FastAPI app
EXPOSE 5000  # MLflow tracking UI


# Start FastAPI using Uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
