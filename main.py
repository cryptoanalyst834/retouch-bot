import logging
import os
import json
import asyncio
import csv
import cv2
import requests
import numpy as np
from io import BytesIO
from PIL import Image
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
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

RETOUCH_WAITING_FOR_IMAGE = 1
RETOUCH_WAITING_FOR_OPTION = 2
MAX_FREE_RETOUCHES = 5
ADMIN_IDS = [743050845]  # твой Telegram user_id
USERS_FILE = "users.json"

if not os.path.exists(USERS_FILE):
    with open(USERS_FILE, "w") as f: f.write("{}")

with open(USERS_FILE, "r") as f:
    users_data = json.load(f)

def save_users(): 
    with open(USERS_FILE, "w") as f: json.dump(users_data, f)

def get_user(user_id): 
    uid = str(user_id)
    if uid not in users_data:
        users_data[uid] = {"count": 0, "is_pro": False}
        save_users()
    return users_data[uid]

def increment_count(user_id): get_user(user_id)["count"] += 1; save_users()
def set_pro(user_id, value=True): get_user(user_id)["is_pro"] = value; save_users()
def reset_count(user_id): get_user(user_id)["count"] = 0; save_users()

# === ОБРАБОТКА ===
def adjust_brightness_contrast(image, brightness=30, contrast=0):
    return cv2.convertScaleAbs(image, alpha=(contrast + 127) / 127, beta=brightness)

def remove_noise(image): return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
def correct_color(image): hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV); h, s, v = cv2.split(hsv); v = cv2.equalizeHist(v); return cv2.cvtColor(cv2.merge((h,s,v)), cv2.COLOR_HSV2BGR)
def skin_retouch(image): return cv2.bilateralFilter(image, 9, 75, 75)
def sharpness(image): return cv2.filter2D(image, -1, np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]))

def full_process(image):
    image = adjust_brightness_contrast(image)
    image = skin_retouch(image)
    image = remove_noise(image)
    image = correct_color(image)
    image = sharpness(image)
    return image

def merge_images(img1, img2):
    im1 = Image.fromarray(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    im2 = Image.fromarray(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
    new_im = Image.new('RGB', (im1.width + im2.width, max(im1.height, im2.height)))
    new_im.paste(im1, (0, 0)); new_im.paste(im2, (im1.width, 0))
    out = BytesIO(); new_im.save(out, format="JPEG"); out.seek(0)
    return out

def neural_retouch(image_path):
    api_key = os.getenv("DEEPAI_API_KEY")
    resp = requests.post("https://api.deepai.org/api/torch-srgan",
                         files={"image": open(image_path, "rb")},
                         headers={"api-key": api_key})
    url = resp.json().get("output_url")
    return requests.get(url).content if url else None

# === HANDLERS ===
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Привет! Отправь /retouch, чтобы начать ✨")

async def retouch(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("📎 Отправь фото файлом.\n(Не сжимай его в Telegram!)")
    return RETOUCH_WAITING_FOR_IMAGE

async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    user = get_user(uid)

    if not user["is_pro"] and user["count"] >= MAX_FREE_RETOUCHES:
        await update.message.reply_text("Вы использовали лимит. Получите Pro-доступ 💎")
        return ConversationHandler.END

    if update.message.document:
        file = await update.message.document.get_file()
        bts = BytesIO(); await file.download_to_memory(out=bts)
        img = cv2.imdecode(np.frombuffer(bts.getvalue(), np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            await update.message.reply_text("Невозможно обработать изображение.")
            return ConversationHandler.END

        path = f"temp_{update.message.message_id}.jpg"
        cv2.imwrite(path, img)
        context.user_data["image"] = img
        context.user_data["path"] = path

        buttons = [
            [InlineKeyboardButton("Лайт ✨", callback_data="light")],
            [InlineKeyboardButton("Бьюти 💄", callback_data="beauty")],
            [InlineKeyboardButton("Про 🎯", callback_data="pro")],
            [InlineKeyboardButton("Нейроретушь 🧠", callback_data="neuro")],
        ]
        await update.message.reply_text("Выбери режим:", reply_markup=InlineKeyboardMarkup(buttons))
        return RETOUCH_WAITING_FOR_OPTION

async def apply_option(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    uid = query.from_user.id
    data = query.data
    img = context.user_data["image"]
    path = context.user_data["path"]

    try:
        if data == "light":
            result = correct_color(adjust_brightness_contrast(img))
        elif data == "beauty":
            result = sharpness(remove_noise(skin_retouch(img)))
        elif data == "pro":
            result = full_process(img)
        elif data == "neuro":
            if not get_user(uid)["is_pro"]:
                await query.edit_message_text("Нейросеть доступна только для Pro 💎")
                return ConversationHandler.END
            content = neural_retouch(path)
            result = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_COLOR)
        else:
            return ConversationHandler.END

        increment_count(uid)
        merged = merge_images(img, result)
        await query.message.reply_photo(merged, caption=f"Готово ✅")
        await query.edit_message_text("Обработка завершена.")
    except Exception as e:
        logger.error(e)
        await query.edit_message_text("Ошибка.")
    finally:
        if os.path.exists(path): os.remove(path)
        context.user_data.clear()
    return ConversationHandler.END

# === ADMIN ===
async def admin(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id not in ADMIN_IDS:
        return await update.message.reply_text("Нет доступа ❌")
    rep = [f"{uid}: Pro={d['is_pro']} Обработки={d['count']}" for uid,d in users_data.items()]
    await update.message.reply_text("\n".join(rep))

async def setpro(update: Update, context: ContextTypes.DEFAULT_TYPE):
    set_pro(context.args[0], True)
    await update.message.reply_text("✅ Pro выдан")

async def revokepro(update: Update, context: ContextTypes.DEFAULT_TYPE):
    set_pro(context.args[0], False)
    await update.message.reply_text("❌ Pro убран")

async def resetcount(update: Update, context: ContextTypes.DEFAULT_TYPE):
    reset_count(context.args[0])
    await update.message.reply_text("🔄 Сброс выполнен")

async def export_users(update: Update, context: ContextTypes.DEFAULT_TYPE):
    with open("users.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["user_id", "is_pro", "count"])
        for uid, d in users_data.items():
            writer.writerow([uid, d["is_pro"], d["count"]])
    with open("users.csv", "rb") as f:
        await update.message.reply_document(f)
    os.remove("users.csv")

# === MAIN ===
WEBHOOK_PATH = "/webhook"
WEBHOOK_URL = os.getenv("WEBHOOK_BASE") + WEBHOOK_PATH

async def main():
    app = Application.builder().token(os.getenv("BOT_TOKEN")).build()

    conv = ConversationHandler(
        entry_points=[CommandHandler("retouch", retouch)],
        states={
            RETOUCH_WAITING_FOR_IMAGE: [MessageHandler(filters.Document.IMAGE | filters.PHOTO, handle_photo)],
            RETOUCH_WAITING_FOR_OPTION: [CallbackQueryHandler(apply_option)],
        },
        fallbacks=[]
    )

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("admin", admin))
    app.add_handler(CommandHandler("setpro", setpro))
    app.add_handler(CommandHandler("revokepro", revokepro))
    app.add_handler(CommandHandler("resetcount", resetcount))
    app.add_handler(CommandHandler("exportusers", export_users))
    app.add_handler(conv)

    await app.bot.set_webhook(url=WEBHOOK_URL)
    await app.updater.start_webhook(
        listen="0.0.0.0",
        port=int(os.getenv("PORT", 8000))
    )
    await app.updater.wait()

if __name__ == '__main__':
    asyncio.run(main())
