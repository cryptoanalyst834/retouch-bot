import logging
import cv2
import numpy as np
from io import BytesIO
from PIL import Image
import os
import json
import requests
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    ConversationHandler,
    ContextTypes,
    filters,
)
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

RETOUCH_WAITING_FOR_IMAGE = 1
RETOUCH_WAITING_FOR_OPTION = 2
ALLOWED_EXTENSIONS = ['.jpg', '.jpeg', '.heic']
MAX_FREE_RETOUCHES = 5
ADMIN_IDS = [743050845]

INSTRUCTIONS_TEXT = (
    "📎 Отправьте фото *файлом*, не сжимая изображение.\n\n"
    "📱 Телефон: Скрепка → Файл → Галерея → Отправить.\n"
    "🖥 ПК: Скрепка → Выбрать файл → Убрать галочку 'Сжать изображение' → Отправить.\n\n"
    "Поддерживаются форматы JPG/JPEG/HEIC."
)

USERS_FILE = "users.json"
def load_users():
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_users(data):
    with open(USERS_FILE, 'w') as f:
        json.dump(data, f)

users_data = load_users()

def get_user_data(user_id):
    user_id = str(user_id)
    if user_id not in users_data:
        users_data[user_id] = {"count": 0, "is_pro": False}
        save_users(users_data)
    return users_data[user_id]

def increment_user_count(user_id):
    user = get_user_data(user_id)
    user["count"] += 1
    save_users(users_data)

def user_has_access(user_id):
    user = get_user_data(user_id)
    return user["is_pro"] or user["count"] < MAX_FREE_RETOUCHES

def user_is_pro(user_id):
    return get_user_data(user_id).get("is_pro", False)

def set_user_pro(user_id, value: bool):
    user = get_user_data(user_id)
    user["is_pro"] = value
    save_users(users_data)

def reset_user_count(user_id):
    user = get_user_data(user_id)
    user["count"] = 0
    save_users(users_data)

# === Telegram Admin Handlers ===
async def admin(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if user_id not in ADMIN_IDS:
        await update.message.reply_text("Нет доступа ❌")
        return

    report = [f"Всего пользователей: {len(users_data)}"]
    for uid, data in users_data.items():
        report.append(f"{uid}: Pro={data['is_pro']}, Обработки={data['count']}")
    await update.message.reply_text("\n".join(report))

async def setpro(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if user_id not in ADMIN_IDS:
        await update.message.reply_text("Нет доступа ❌")
        return
    if not context.args:
        await update.message.reply_text("Использование: /setpro user_id")
        return
    try:
        target_id = int(context.args[0])
        set_user_pro(target_id, True)
        await update.message.reply_text(f"Пользователю {target_id} выдан Pro ✅")
    except Exception as e:
        await update.message.reply_text(f"Ошибка: {e}")

async def revokepro(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if user_id not in ADMIN_IDS:
        await update.message.reply_text("Нет доступа ❌")
        return
    if not context.args:
        await update.message.reply_text("Использование: /revokepro user_id")
        return
    try:
        target_id = int(context.args[0])
        set_user_pro(target_id, False)
        await update.message.reply_text(f"Pro-доступ пользователя {target_id} удалён ❌")
    except Exception as e:
        await update.message.reply_text(f"Ошибка: {e}")

async def resetcount(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if user_id not in ADMIN_IDS:
        await update.message.reply_text("Нет доступа ❌")
        return
    if not context.args:
        await update.message.reply_text("Использование: /resetcount user_id")
        return
    try:
        target_id = int(context.args[0])
        reset_user_count(target_id)
        await update.message.reply_text(f"Счётчик пользователя {target_id} сброшен 🔄")
    except Exception as e:
        await update.message.reply_text(f"Ошибка: {e}")
