"""
LESSON 7 (part 2) — USING the shipped spam filter
=================================================
This is the "production" side: no training code at all. Load the saved
pipeline and classify new messages — exactly how a deployed model works.

Run train.py first. Then:
    predict.py                      -> demo messages
    predict.py "your message here"  -> classify your own message
"""

import sys
import joblib
from pathlib import Path

HERE = Path(__file__).parent
model = joblib.load(HERE / "spam_filter.joblib")  # vectorizer + model in one

demo_messages = [
    "URGENT winner! Claim your free vacation now at bit.ly/xy",
    "hey can you send me the notes from today's class?",
    "You have won $5000, verify your bank account immediately",
    "running late, save me a seat",
]
messages = sys.argv[1:] if len(sys.argv) > 1 else demo_messages

for msg in messages:
    label = model.predict([msg])[0]
    # confidence for the predicted class:
    proba = model.predict_proba([msg]).max()
    flag = "🚫 SPAM" if label == "spam" else "✅ ham "
    print(f"{flag} ({proba:.0%} sure)  {msg}")

# =====================================================================
# TRY IT YOURSELF:
#  - ../.venv/bin/python predict.py "free free free click here to win"
#  - Try to FOOL it: write spam without any typical spam words. This is
#    the real cat-and-mouse game spam filter engineers play daily.
# =====================================================================
