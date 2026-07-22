FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN groupadd --system fraud && useradd --system --gid fraud --home /app fraud
COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY schemas ./schemas
RUN mkdir -p /app/models && chown -R fraud:fraud /app

USER fraud
CMD ["python", "-m", "src.api.outbox_relay", "--help"]
