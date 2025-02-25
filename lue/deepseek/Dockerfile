FROM python:3.9-slim

WORKDIR /app

RUN apt update && apt install -y git && rm -rf /var/lib/apt/lists/*

RUN pip install torch transformers fastapi uvicorn

COPY . /app
CMD ["python", "server.py"]

