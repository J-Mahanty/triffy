FROM python:3.12-slim

# Prevent Python from creating .pyc files
ENV PYTHONDONTWRITEBYTECODE=1 

# Prevent Python from adding buffer
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install dependencies
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt


COPY discordBot/ ./discordBot/
COPY route_engine/ ./route_engine/
COPY data/ ./data/

CMD ["python", "discordBot/main.py"]
