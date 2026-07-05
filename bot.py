import os
import pickle

from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, ContextTypes, filters

from preprocess import preprocess

# Load AI model
model = pickle.load(open("spam_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

TOKEN = "8692527290:AAFi94fngLj-ahjkrVUBl80urkdrwZsOO1k"

TOKEN = os.environ["BOT_TOKEN"]

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    welcome_text = (
        "Welcome to AI Spam Detector Bot.\n\n"
        "Send me any message and I'll analyze it using a trained "
        "machine learning model to tell you whether it's spam or not, "
        "along with a confidence score.\n\n"
        "Just type or paste a message to get started — no commands needed."
    )
    await update.message.reply_text(welcome_text)

async def detect_spam(update: Update, context: ContextTypes.DEFAULT_TYPE):

    # Get user's message
    user_message = update.message.text

    # Preprocess message
    cleaned_message = preprocess(user_message)

    # Convert text to numbers
    message_vec = vectorizer.transform([cleaned_message])

    # Predict
    prediction = model.predict(message_vec)[0]

    # Confidence
    probabilities = model.predict_proba(message_vec)[0]

    print("=" * 40)
    print("Original:", user_message)
    print("Cleaned :", cleaned_message)
    print("Prediction:", prediction)
    print("Probabilities:", probabilities)
    print("=" * 40)

    spam_confidence = probabilities[1] * 100
    ham_confidence = probabilities[0] * 100

    # Reply
    if prediction == 1:
        reply = (
            "🤖 AI Spam Detector\n\n"
            "⚠️ Prediction: Spam\n\n"
            f"📊 Confidence: {spam_confidence:.2f}%\n\n"
            "💡 Recommendation:\n"
            "Avoid clicking suspicious links or sharing personal information."
        )
    else:
        reply = (
            "🤖 AI Spam Detector\n\n"
            "✅ Prediction: Not Spam\n\n"
            f"📊 Confidence: {ham_confidence:.2f}%\n\n"
            "💡 Recommendation:\n"
            "This message appears safe based on the AI model."
        )

    await update.message.reply_text(reply)

app = Application.builder().token(TOKEN).build()

app.add_handler(CommandHandler("start", start))
app.add_handler(
    MessageHandler(filters.TEXT & ~filters.COMMAND, detect_spam)
)

print("🤖 AI Spam Detector Bot is running...")

app.run_polling()
