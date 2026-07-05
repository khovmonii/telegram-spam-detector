import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
import pickle

from preprocess import preprocess

# Read csv file
df = pd.read_csv("data/cleaned_spam_data.csv")

# Remove rows where clean_message is missing
df = df.dropna(subset=["clean_message"])

# Split data
X = df['clean_message']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# TF-IDF
vectorizer = TfidfVectorizer()

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Evaluate
y_pred = model.predict(X_test_vec)

print("Accuracy:", accuracy_score(y_test, y_pred))


def predict_message(msg):
    cleaned_msg = preprocess(msg)
    msg_vec = vectorizer.transform([cleaned_msg])

    # Probability of each class
    probs = model.predict_proba(msg_vec)[0]

    spam_prob = probs[1] * 100
    ham_prob = probs[0] * 100

    if spam_prob > ham_prob:
        return f"⚠ Spam Detected ({spam_prob:.2f}% confidence)"
    else:
        return f"✅ Not Spam ({ham_prob:.2f}% confidence)"
    
print(predict_message("Win a free iPhone now!"))
print(predict_message("Are we meeting tomorrow?"))

# Save model and vectorizer for Telegram bot
pickle.dump(model, open("spam_model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("Model saved successfully!")

print(predict_message("Win a free iPhone now!"))
print(predict_message("Congratulations you've free iPhone click"))
print(predict_message("Are we meeting tomorrow?"))

