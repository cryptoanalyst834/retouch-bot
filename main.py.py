﻿import logging
import cv2
import numpy as np
from io import BytesIO
from PIL import Image
import os
from telegram import (
    Update,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
)
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    ConversationHandler,
    ContextTypes,
    filters,
)

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

RETOUCH_WAITING_FOR_IMAGE = 1
RETOUCH_WAITING_FOR_OPTION = 2

ALLOWED_EXTENSIONS = ['.jpg', '.jpeg', '.heic']
user_processing_counts = {}
MAX_FREE_RETOUCHES = 5

INSTRUCTIONS_TEXT = (
    "📎 Отправьте фото *файлом*, не сжимая изображение.\n\n"
    "📱 Как отправить с телефона:\n"
    "1. Нажмите на скрепку → Файл → Выбрать из Галереи → Отправить.\n\n"
    "🖥 С ПК:\n"
    "1. Скрепка → Выбрать файл → Снять галочку 'Сжать изображение' → Отправить.\n\n"
    "Поддерживаются только JPG/JPEG/HEIC."
)

# === Обработка изображений ===
def adjust_brightness_contrast(image, brightness=30, contrast=0):
    beta = brightness
    alpha = (contrast + 127) / 127
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

def remove_noise(image):
    return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)

def correct_color_exposure(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v_eq = cv2.equalizeHist(v)
    hsv_eq = cv2.merge((h, s, v_eq))
    return cv2.cvtColor(hsv_eq, cv2.COLOR_HSV2BGR)

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
    total_width = im1.width + im2.width
    max_height = max(im1.height, im2.height)
    new_im = Image.new('RGB', (total_width, max_height))
    new_im.paste(im1, (0, 0))
    new_im.paste(im2, (im1.width, 0))
    output = BytesIO()
    new_im.save(output, format='JPEG')
    output.seek(0)
    return output

# === Telegram Handlers ===
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Привет! Я бот EasyRetouch ✨\n\nОтправь команду /retouch для обработки фото."
    )

async def retouch_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    await update.message.reply_text(INSTRUCTIONS_TEXT, parse_mode="Markdown")
    return RETOUCH_WAITING_FOR_IMAGE

async def retouch_photo_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    user_id = update.effective_user.id
    if user_processing_counts.get(user_id, 0) >= MAX_FREE_RETOUCHES:
        await update.message.reply_text(
            "Вы использовали 5 бесплатных улучшений.\nХотите продолжить? 🔓 Получите Pro-доступ."
        )
        return ConversationHandler.END

    if update.message.document:
        file_name = update.message.document.file_name.lower()
        if not any(file_name.endswith(ext) for ext in ALLOWED_EXTENSIONS):
            await update.message.reply_text(INSTRUCTIONS_TEXT)
            return ConversationHandler.END

        file = await update.message.document.get_file()
        file_bytes = BytesIO()
        await file.download_to_memory(out=file_bytes)
        data = np.frombuffer(file_bytes.getvalue(), dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is None:
            await update.message.reply_text("Ошибка при декодировании изображения.")
            return ConversationHandler.END

        context.user_data["original_image"] = img

        keyboard = [
            [InlineKeyboardButton("Лайт ✨", callback_data="preset:light")],
            [InlineKeyboardButton("Бьюти 💄", callback_data="preset:beauty")],
            [InlineKeyboardButton("Про 🎯", callback_data="preset:pro")],
        ]
        await update.message.reply_text("Выбери тип ретуши:", reply_markup=InlineKeyboardMarkup(keyboard))
        return RETOUCH_WAITING_FOR_OPTION
    else:
        await update.message.reply_text(INSTRUCTIONS_TEXT)
        return ConversationHandler.END

async def retouch_option_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await query.answer()
    user_id = query.from_user.id

    original = context.user_data.get("original_image")
    if original is None:
        await query.edit_message_text("Файл не найден. Попробуй снова.")
        return ConversationHandler.END

    option = query.data.split(":")[1]
    if option == "light":
        result = correct_color_exposure(adjust_brightness_contrast(original))
    elif option == "beauty":
        result = enhance_sharpness(remove_noise(skin_retouch(original)))
    elif option == "pro":
        result = full_process(original)
    else:
        await query.edit_message_text("Неверный выбор.")
        return ConversationHandler.END

    user_processing_counts[user_id] = user_processing_counts.get(user_id, 0) + 1

    merged = merge_images(original, result)
    await query.message.reply_photo(merged, caption=f"Результат обработки: {option.title()} режим")
    await query.edit_message_text("Обработка завершена ✅")
    context.user_data.clear()
    return ConversationHandler.END

async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    await update.message.reply_text("Операция отменена ❌")
    return ConversationHandler.END

# === Webhook ===
WEBHOOK_PATH = "/webhook"
WEBHOOK_URL = f"{os.getenv('WEBHOOK_BASE')}{WEBHOOK_PATH}"

def main():
    app = Application.builder().token(os.getenv("BOT_TOKEN")).build()

    conv_handler = ConversationHandler(
        entry_points=[CommandHandler("retouch", retouch_start)],
        states={
            RETOUCH_WAITING_FOR_IMAGE: [
                MessageHandler(filters.Document.IMAGE | filters.PHOTO, retouch_photo_handler)
            ],
            RETOUCH_WAITING_FOR_OPTION: [
                CallbackQueryHandler(retouch_option_handler, pattern="^preset:")
            ]
        },
        fallbacks=[CommandHandler("cancel", cancel)]
    )

    app.add_hndler(CommandHandler("start", start))
    app.add_handler(conv_handler)

    app.run_webhook(
        listen="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        webhook_path=WEBHOOK_PATH,
        webhook_url=WEBHOOK_URL
    )

if __name__ == '__main__':
    main()
