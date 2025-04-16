FROM python:3.11-slim

WORKDIR /app

COPY . .

# Устанавливаем зависимости для OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev

# Устанавливаем Python-зависимости
RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "main.py"]
