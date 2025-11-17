FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y build-essential curl && rm -rf /var/lib/apt/lists/*

COPY docker/encoders/gcn/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY docker/encoders/gcn/app.py /app/app.py

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]