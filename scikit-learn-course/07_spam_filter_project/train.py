"""
LESSON 7 — FINAL PROJECT: SMS Spam Filter (end-to-end)
======================================================
Real-life problem: Build the thing that sits inside every phone and email
service — a filter that reads a message and decides: SPAM or HAM (normal)?

This lesson ties EVERYTHING together, plus one new superpower:
  - TfidfVectorizer: turning raw TEXT into numeric features
    (each word becomes a feature; rare-but-distinctive words weigh more)
  - MultinomialNB (Naive Bayes): fast, classic text classifier
  - Saving the WHOLE pipeline (vectorizer + model) as one file,
    then using it from a separate predict.py — like a real product.

Run:  train.py  first, then  predict.py
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

HERE = Path(__file__).parent

# =====================================================================
# 1. THE DATA — labeled example messages
# =====================================================================
# In real life you'd load 5000+ labeled SMS from a CSV (search for the
# "SMS Spam Collection" dataset). We build a smaller inline set from
# realistic templates so this script has no downloads.
spam_templates = [
    "WINNER!! You have been selected to receive a {prize} prize. Call now!",
    "URGENT! Your account will be suspended. Verify at {link} immediately",
    "Congratulations! Claim your FREE {prize} today. Limited offer, click {link}",
    "You won a lottery of {prize}! Send your bank details to claim",
    "FINAL NOTICE: pay ${amount} now to avoid legal action. Call this number",
    "Hot singles in your area! Visit {link} for FREE access",
    "Get a loan of ${amount} instantly, no credit check! Reply YES",
    "Your package is held. Pay ${amount} customs fee at {link}",
]
ham_templates = [
    "Hey, are we still on for lunch at {time}?",
    "Can you pick up milk on the way home?",
    "The meeting got moved to {time}, see you there",
    "Happy birthday!! Hope you have a great day",
    "I'll be {mins} minutes late, sorry, traffic is terrible",
    "Did you finish the report? Boss is asking about it",
    "Mom called, she wants us to visit this weekend",
    "Don't forget the dentist appointment at {time} tomorrow",
    "Can you send me the notes from today's lecture?",
    "Send me the address when you get a chance",
    "Just got home, call me when you're free",
    "The game starts at {time}, want to watch it together?",
    "Thanks for your help today, really appreciate it",
    "Are you coming to class tomorrow or should I take notes for you?",
    "Let me know what time works for you",
    "Forgot my charger at your place, I'll grab it later",
]

rng = np.random.default_rng(42)
def fill(t):
    return (t.replace("{prize}", rng.choice(["$1000", "iPhone", "vacation", "gift card"]))
             .replace("{link}", rng.choice(["bit.ly/x2f", "win-now.co", "secure-verify.net"]))
             .replace("{amount}", str(rng.integers(100, 5000)))
             .replace("{time}", rng.choice(["noon", "3pm", "6:30", "9am"]))
             .replace("{mins}", str(rng.integers(5, 40))))

messages = [fill(t) for t in spam_templates for _ in range(25)] + \
           [fill(t) for t in ham_templates for _ in range(25)]
labels = ["spam"] * (25 * len(spam_templates)) + ["ham"] * (25 * len(ham_templates))

df = pd.DataFrame({"message": messages, "label": labels}).sample(frac=1, random_state=1)
print(f"{len(df)} messages ({(df['label'] == 'spam').mean():.0%} spam)")
print(df.head(4).to_string(index=False), "\n")

X_train, X_test, y_train, y_test = train_test_split(
    df["message"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
)

# =====================================================================
# 2. TEXT -> NUMBERS + MODEL, in one pipeline
# =====================================================================
# TfidfVectorizer builds a vocabulary from training messages and turns
# each message into a vector of word weights. Words like "winner",
# "urgent", "free" will end up highly indicative of spam.
model = Pipeline([
    ("tfidf", TfidfVectorizer(lowercase=True, stop_words="english")),
    ("classifier", MultinomialNB()),
])
model.fit(X_train, y_train)

print("Performance on unseen messages:")
print(classification_report(y_test, model.predict(X_test)))

# Peek inside: which words scream "spam" the loudest?
vocab = model.named_steps["tfidf"].get_feature_names_out()
weights = model.named_steps["classifier"].feature_log_prob_
spam_idx = list(model.named_steps["classifier"].classes_).index("spam")
spamminess = weights[spam_idx] - weights[1 - spam_idx]
top = pd.Series(spamminess, index=vocab).nlargest(10)
print("Top 10 spammiest words the model discovered on its own:")
print(", ".join(top.index), "\n")

# =====================================================================
# 3. SHIP IT — one file contains vectorizer + trained model
# =====================================================================
joblib.dump(model, HERE / "spam_filter.joblib")
print(f"Saved -> {HERE / 'spam_filter.joblib'}")
print("Now run predict.py to use it like a real service!")
