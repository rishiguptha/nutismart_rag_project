# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# --- Add this block to install build tools ---
# hadolint ignore=DL3008 DL3015
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
 && rm -rf /var/lib/apt/lists/*
# build-essential includes compiler (g++), make etc.
# cmake is sometimes needed by C++ projects during build configuration.
# --no-install-recommends keeps the layer smaller
# rm -rf cleans up apt cache afterwards
# --- End of added block ---

# Copy the requirements file into the container first to leverage Docker cache
COPY ./requirements.txt /app/requirements.txt

# Install any needed packages specified in requirements.txt
# (Now the build tools are available for chroma-hnswlib)
RUN pip install --no-cache-dir --upgrade pip -r requirements.txt

# Copy the application code into the container
COPY ./app /app/app

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Define environment variables (optional, can be set at runtime too)
ENV CHROMADB_PATH=/app/vector_store
ENV PYTHONUNBUFFERED=1

# Run main.py when the container launches
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]