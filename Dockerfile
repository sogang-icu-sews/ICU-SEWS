FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src

WORKDIR /app

COPY requirements-deploy.txt .
RUN pip install --no-cache-dir -r requirements-deploy.txt

COPY src/ ./src/
COPY scripts/start.sh ./scripts/start.sh
RUN chmod +x ./scripts/start.sh

EXPOSE 10000

CMD ["bash", "scripts/start.sh"]
