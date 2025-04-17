import os
import json
import csv
import logging
import cv2
import requests
import numpy as np
from io import BytesIO
from PIL import Image
from dotenv import load_dotenv
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

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

RETOUCH_WAITING_FOR_IMAGE = 1
RETOUCH_WAITING_FOR_OPTION = 2
ADMIN_IDS = [743050845]
MAX_FREE_RETOUCHES = 5
USERS_FILE = "users.json"

if not os.path.exists(USERS_FILE):
    with open(USERS_FILE, "w") as f: f.write("{}")
with open(USERS_FILE, "r") as f: users_data = json.load(f)

def save_users(): 
    with open(USERS_FILE, "w") as f: json.dump(users_data, f)
def get_user(uid):
    uid = str(uid)
    if uid not in users_data:
        users_data[uid] = {"count": 0, "is_pro": False}
        save_users()
    return users_data[uid]
def increment(uid): get_user(uid)["count"] += 1; save_users()
def set_pro(uid, val=True): get_user(uid)["is_pro"] = val; save_users()
def reset_count(uid): get_user(uid)["count"] = 0; save_users()

def correct_color(image): hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV); h,s,v=cv2.split(hsv);v=cv2.equalizeHist(v); return cv2.cvtColor(cv2.merge((h,s,v)), cv2.COLOR_HSV2BGR)
def brightness(image): return cv2.convertScaleAbs(image, alpha=1.3, beta=20)
def skin(image): return cv2.bilateralFilter(image, 9, 75, 75)
def noise(image): return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
def sharp(image): return cv2.filter2D(image, -1, np.array([[0,-1,0],[-1,5,-1],[0,-1,0]]))
def full_process(image): return sharp(noise(skin(correct_color(brightness(image)))))

def merge(img1, img2):
    i1 = Image.fromarray(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    i2 = Image.fromarray(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
    w, h = i1.width + i2.width, max(i1.height, i2.height)
    new_im = Image.new('RGB', (w, h))
    new_im.paste(i1, (0, 0)); new_im.paste(i2, (i1.width, 0))
    out = BytesIO(); new_im.save(out, format="JPEG"); out.seek(0); return out

def neural(path):
    api = os.getenv("DEEPAI_API_KEY")
    res = requests.post("https://api.deepai.org/api/torch-srgan",
                        files={"image": open(path, "rb")},
                        headers={"api-key": api})
    url = res.json().get("output_url")
    return requests.get(url).content if url else None

async def start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Привет! Отправь /retouch, чтобы начать.")

async def retouch(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("📎 Отправь фото *файлом* без сжатия.")
    return RETOUCH_WAITING_FOR_IMAGE

async def handle_photo(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
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
            await update.message.reply_text("Ошибка при загрузке.")
            return ConversationHandler.END

        path = f"temp_{update.message.message_id}.jpg"
        cv2.imwrite(path, img)
        ctx.user_data["img"] = img
        ctx.user_data["path"] = path

        kb = [[InlineKeyboardButton("Лайт ✨", callback_data="light")],
              [InlineKeyboardButton("Бьюти 💄", callback_data="beauty")],
              [InlineKeyboardButton("Про 🎯", callback_data="pro")],
              [InlineKeyboardButton("Нейроретушь 🧠", callback_data="neuro")]]
        await update.message.reply_text("Выбери режим:", reply_markup=InlineKeyboardMarkup(kb))
        return RETOUCH_WAITING_FOR_OPTION

async def apply_option(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    uid = q.from_user.id
    mode = q.data
    img = ctx.user_data["img"]
    path = ctx.user_data["path"]

    try:
        if mode == "light":
            result = correct_color(brightness(img))
        elif mode == "beauty":
            result = sharp(noise(skin(img)))
        elif mode == "pro":
            result = full_process(img)
        elif mode == "neuro":
            if not get_user(uid)["is_pro"]:
                await q.edit_message_text("Нейросеть доступна только для Pro 💎")
                return ConversationHandler.END
            content = neural(path)
            result = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_COLOR)
        else:
            return ConversationHandler.END

        increment(uid)
        m = merge(img, result)
        await q.message.reply_photo(m, caption="Готово ✅")
        await q.edit_message_text("Обработка завершена.")
    except Exception as e:
        logger.error(e)
        await q.edit_message_text("Произошла ошибка.")
    finally:
        if os.path.exists(path): os.remove(path)
        ctx.user_data.clear()
    return ConversationHandler.END

# === АДМИН ===
async def admin(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id not in ADMIN_IDS:
        return await update.message.reply_text("Нет доступа ❌")
    text = [f"{uid}: Pro={d['is_pro']} Обработки={d['count']}" for uid,d in users_data.items()]
    await update.message.reply_text("\n".join(text))

async def setpro(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    set_pro(ctx.args[0], True)
    await update.message.reply_text("✅ Pro выдан")

async def revokepro(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    set_pro(ctx.args[0], False)
    await update.message.reply_text("❌ Pro убран")

async def resetcount(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    reset_count(ctx.args[0])
    await update.message.reply_text("🔄 Сброс")

async def export_users(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    with open("users.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["user_id", "is_pro", "count"])
        for uid, d in users_data.items():
            writer.writerow([uid, d["is_pro"], d["count"]])
    with open("users.csv", "rb") as f:
        await update.message.reply_document(f)
    os.remove("users.csv")

def main():
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

    app.run_polling()

if __name__ == '__main__':
    main()
