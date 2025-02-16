# Use Debian-based slim Python image
FROM python:3.11-slim


# Set the working directory in the container
WORKDIR /app

# RUN mkdir -p /data

RUN pip install --upgrade pip setuptools wheel
ARG AIPROXY_TOKEN
ENV AIPROXY_TOKEN=${AIPROXY_TOKEN}

# Install system dependencies required for libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    make \
    libffi-dev \
    musl-dev \
    libopenblas-dev \
    sqlite3 \
    libsqlite3-dev \
    libmagic-dev \
    tesseract-ocr \
    curl \
    git \
    nodejs \
    wget \
    npm \
    && rm -rf /var/lib/apt/lists/*

# Install Prettier globally
RUN npm install -g prettier

# Copy the FastAPI application code into the container
COPY  tds-project-1/app.py .
COPY tds-project-1/faiss_index.index .

# Install uv
RUN pip install uv

# Expose the port that the app runs on
EXPOSE 8000

# Command to run the application
CMD ["uv", "run", "app.py"]