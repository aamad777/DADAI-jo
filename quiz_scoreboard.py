
import json
import os
from datetime import datetime
import streamlit as st

LOG_FILE = "quiz_log.json"

def log_score(name, score):
    entry = {
        "name": name,
        "score": score,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = []

    data.append(entry)
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def show_scoreboard():
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        st.subheader("📊 Quiz Scoreboard")
        for entry in reversed(data[-10:]):
            st.markdown(f"**👧 {entry['name']}** — {entry['score']}/5 ⭐ (_{entry['timestamp']}_)")
    else:
        st.info("No scores yet. Be the first to take a quiz!")
