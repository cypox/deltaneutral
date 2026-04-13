FROM python:3.11-slim AS base

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
COPY src/ src/

RUN pip install --no-cache-dir -e ".[backtest]"

# Create data and log directories
RUN mkdir -p data logs

# Non-root user
RUN useradd --create-home appuser
USER appuser

ENTRYPOINT ["tradingbot"]
CMD ["run", "--config", "config/config.yaml"]
