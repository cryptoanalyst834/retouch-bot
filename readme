# 📸 EasyRetouch Telegram Bot

Бот для обработки и ретуши фотографий в Telegram с использованием Python, OpenCV и Telegram Bot API.

---

## 🚀 Функции

- Ретушь кожи, шумоподавление, цветокоррекция, улучшение резкости
- 3 режима обработки: Лайт, Бьюти, Про
- Поддержка Webhook для продакшена (Railway, Render)
- Поддержка ограничений на количество бесплатных обработок

---

## 📁 Установка локально

```bash
git clone https://github.com/ВАШ_ЛОГИН/retouch-bot.git
cd retouch-bot
cp .env.example .env
# Заполните .env
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

---

## ⚙️ Переменные окружения `.env`

```env
BOT_TOKEN=your_telegram_bot_token
WEBHOOK_BASE=https://your-railway-url.up.railway.app
```

---

## ☁️ Деплой на Railway

1. Перейдите на [https://railway.app](https://railway.app)
2. Нажмите `New Project` → `Deploy from GitHub repo`
3. Выберите этот репозиторий
4. Railway сам запустит проект по `Dockerfile`
5. Добавьте переменные `BOT_TOKEN` и `WEBHOOK_BASE`

---

## 🐳 Docker

```Dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir -r requirements.txt
CMD ["python", "main.py"]
```

---

## 📜 License
MIT License

---

## 💬 Связь
[Telegram: @yourusername]

