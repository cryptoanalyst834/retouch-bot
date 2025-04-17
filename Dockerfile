FROM python:3.11-slim

WORKDIR /app

COPY . .

# Установка зависимостей для OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev

# Установка Python-зависимостей
RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "main.py"]
