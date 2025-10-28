FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md config.yaml ./
COPY src ./src

RUN pip install --no-cache-dir -e .

COPY data ./data
COPY logs ./logs
COPY reports ./reports

CMD ["lc", "run"]
