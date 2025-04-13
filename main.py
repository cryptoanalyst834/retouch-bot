main_code = '''import logging
import cv2
import numpy as np
from io import BytesIO
from PIL import Image
import os
import json
import requests
import csv
import asyncio
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
    "📎 Отправьте фото *файлом*, не сжимая изображение.\\n\\n"
    "📱 Телефон: Скрепка → Файл → Галерея → Отправить.\\n"
    "🖥 ПК: Скрепка → Выбрать файл → Убрать галочку 'Сжать изображение' → Отправить.\\n\\n"
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

def adjust_brightness_contrast(image, brightness=30, contrast=0):
    return cv2.convertScaleAbs(image, alpha=(contrast + 127) / 127, beta=brightness)

def remove_noise(image):
    return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)

def correct_color_exposure(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v_eq = cv2.equalizeHist(v)
    return cv2.cvtColor(cv2.merge((h, s, v_eq)), cv2.COLOR_HSV2BGR)

def skin_retouch(image):
    return cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)

def enhance_sharpness(image):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    return cv2.filter2D(image, -1, kernel)

def full_process(image):
    image = adjust_brightness_contrast(image)
    image = skin_retouch(image)
    image = remove_noise(image)
    image = correct_color_exposure(image)
    image = enhance_sharpness(image)
    return image

def merge_images(img1, img2):
    im1 = Image.fromarray(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    im2 = Image.fromarray(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
    new_im = Image.new('RGB', (im1.width + im2.width, max(im1.height, im2.height)))
    new_im.paste(im1, (0, 0))
    new_im.paste(im2, (im1.width, 0))
    output = BytesIO()
    new_im.save(output, format='JPEG')
    output.seek(0)
    return output

def neural_retouch_deepai(image_path: str) -> bytes:
    api_key = os.getenv("DEEPAI_API_KEY")
    response = requests.post(
        "https://api.deepai.org/api/torch-srgan",
        files={"image": open(image_path, "rb")},
        headers={"api-key": api_key}
    )
    result_url = response.json().get("output_url")
    if result_url:
        return requests.get(result_url).content
    else:
        raise RuntimeError("Ошибка нейросети")

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Привет! Я бот EasyRetouch ✨\\nИспользуй /retouch, чтобы начать обработку фото.")

async def retouch_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(INSTRUCTIONS_TEXT, parse_mode="Markdown")
    return RETOUCH_WAITING_FOR_IMAGE

async def retouch_photo_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if not user_has_access(user_id):
        await update.message.reply_text("Вы использовали лимит. Получите Pro-доступ 💎")
        return ConversationHandler.END

    if update.message.document:
        file = await update.message.document.get_file()
        file_bytes = BytesIO()
        await file.download_to_memory(out=file_bytes)
        img = cv2.imdecode(np.frombuffer(file_bytes.getvalue(), dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            await update.message.reply_text("Ошибка при чтении изображения.")
            return ConversationHandler.END

        file_path = f"temp_{update.message.message_id}.jpg"
        cv2.imwrite(file_path, img)
        context.user_data["original_image"] = img
        context.user_data["file_path"] = file_path

        keyboard = [
            [InlineKeyboardButton("Лайт ✨", callback_data="preset:light")],
            [InlineKeyboardButton("Бьюти 💄", callback_data="preset:beauty")],
            [InlineKeyboardButton("Про 🎯", callback_data="preset:pro")],
            [InlineKeyboardButton("Нейроретушь 🧠", callback_data="preset:neuro")],
        ]
        await update.message.reply_text("Выбери режим ретуши:", reply_markup=InlineKeyboardMarkup(keyboard))
        return RETOUCH_WAITING_FOR_OPTION

async def retouch_option_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    user_id = query.from_user.id
    mode = query.data.split(":")[1]
    image = context.user_data["original_image"]
    file_path = context.user_data["file_path"]

    try:
        if mode == "light":
            result = correct_color_exposure(adjust_brightness_contrast(image))
        elif mode == "beauty":
            result = enhance_sharpness(remove_noise(skin_retouch(image)))
        elif mode == "pro":
            result = full_process(image)
        elif mode == "neuro":
            if not user_is_pro(user_id):
                await query.edit_message_text("Нейроретушь доступна только для Pro 💎")
                return ConversationHandler.END
            processed_bytes = neural_retouch_deepai(file_path)
            result = cv2.imdecode(np.frombuffer(processed_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        else:
            return ConversationHandler.END

        increment_user_count(user_id)
        merged = merge_images(image, result)
        await query.message.reply_photo(merged, caption=f"Режим: {mode.title()} ✅")
        await query.edit_message_text("Готово! 🎉")
    except Exception as e:
        logger.error(e)
        await query.edit_message_text("Произошла ошибка.")
    finally:
        os.remove(file_path)
        context.user_data.clear()
    return ConversationHandler.END

# Админ-команды
async def admin(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id not in ADMIN_IDS:
        return await update.message.reply_text("Нет доступа ❌")
    report = [f"{uid}: Pro={d['is_pro']} Обработки={d['count']}" for uid, d in users_data.items()]
    await update.message.reply_text("\\n".join(report))

async def setpro(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id not in ADMIN_IDS: return
    uid = context.args[0]
    set_user_pro(uid, True)
    await update.message.reply_text(f"{uid} получил Pro ✅")

async def revokepro(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id not in ADMIN_IDS: return
    uid = context.args[0]
    set_user_pro(uid, False)
    await update.message.reply_text(f"{uid} Pro снят ❌")

async def resetcount(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id not in ADMIN_IDS: return
    uid = context.args[0]
    reset_user_count(uid)
    await update.message.reply_text(f"{uid} сброшен 🔁")

async def export_users(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id not in ADMIN_IDS: return
    with open("users_export.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["user_id", "is_pro", "count"])
        for uid, d in users_data.items():
            writer.writerow([uid, d["is_pro"], d["count"]])
    with open("users_export.csv", "rb") as f:
        await update.message.reply_document(f)
    os.remove("users_export.csv")

# Запуск через Webhook
WEBHOOK_PATH = "/webhook"
WEBHOOK_URL = os.getenv("WEBHOOK_BASE") + WEBHOOK_PATH

async def main():
    app = Application.builder().token(os.getenv("BOT_TOKEN")).build()

    conv = ConversationHandler(
        entry_points=[CommandHandler("retouch", retouch_start)],
        states={
            RETOUCH_WAITING_FOR_IMAGE: [MessageHandler(filters.Document.IMAGE | filters.PHOTO, retouch_photo_handler)],
            RETOUCH_WAITING_FOR_OPTION: [CallbackQueryHandler(retouch_option_handler, pattern="^preset:")],
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
        port=int(os.getenv("PORT", 8000)),
        webhook_path=WEBHOOK_PATH,
    )
    await app.updater.wait()

if __name__ == '__main__':
    asyncio.run(main())
'''

with open("/mnt/data/main.py", "w") as f:
    f.write(main_code)

"/mnt/data/main.py"
