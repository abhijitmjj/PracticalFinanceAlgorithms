# syntax=docker/dockerfile:1

# Base image
FROM python:3.12-slim AS base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100 \
    POETRY_VERSION=1.6.1 \
    POETRY_HOME="/opt/poetry" \
    POETRY_VIRTUALENVS_IN_PROJECT=true \
    POETRY_NO_INTERACTION=1 \
    PYSETUP_PATH="/opt/pysetup" \
    VENV_PATH="/opt/pysetup/.venv"

# Update PATH
ENV PATH="$POETRY_HOME/bin:$VENV_PATH/bin:$PATH"

# Builder stage
FROM base AS builder

# Install build dependencies
RUN apt-get update && \
    apt-get install --no-install-recommends -y curl build-essential && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 -

# Set work directory
WORKDIR $PYSETUP_PATH

# Copy Poetry files
COPY pyproject.toml poetry.lock ./

# Install dependencies
RUN poetry install --no-dev

# Production stage
FROM base AS production

# Copy virtual environment from builder
COPY --from=builder $VENV_PATH $VENV_PATH

# Copy application code
COPY src/ /app/src/
COPY main.py /app/

# Set work directory
WORKDIR /app

# Set entrypoint
CMD ["python", "main.py"]
