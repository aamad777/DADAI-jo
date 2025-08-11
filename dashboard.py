import streamlit as st
import json
import os
from datetime import datetime

LOG_FILE = "qa_log.json"

# Ensure the log file exists
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w") as f:
        json.dump([], f)

def load_qa_log():
    try:
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error loading QA log: {e}")
        return []

def save_qa_log(data):
    try:
        with open(LOG_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        st.error(f"Error saving QA log: {e}")

def render_dashboard_tab():
    st.subheader("📊 Dad's Dashboard")

    password = st.text_input("🔐 Enter dashboard password:", type="password")
    if password != "dad123":
        st.warning("Password required to access dashboard")
        return

    qa_data = load_qa_log()
    if not isinstance(qa_data, list):
        st.error("Log data is not valid")
        return

    filter_text = st.text_input("🔍 Search by name or keyword:")

    # Safe filtering with index
    filtered_data = []
    for i, q in enumerate(qa_data):
        if isinstance(q, dict):
            if (
                filter_text.lower() in q.get("question", "").lower()
                or filter_text.lower() in q.get("answer", "").lower()
                or filter_text.lower() in q.get("name", "").lower()
            ):
                filtered_data.append((i, q))

    for i, entry in filtered_data:
        with st.container():
            st.markdown(f"**👧 Name:** {entry.get('name', 'N/A')}")
            st.markdown(f"**❓ Question:** {entry.get('question', '')}")
            st.markdown(f"**💬 Answer:** {entry.get('answer', '')}")
            st.markdown(f"🕒 _Asked on: {entry.get('timestamp', 'N/A')}_")

            new_answer = st.text_input(f"✏️ Edit Answer", value=entry.get("answer", ""), key=f"edit_{i}")
            if st.button("💾 Save", key=f"save_{i}"):
                qa_data[i]["answer"] = new_answer
                save_qa_log(qa_data)
                st.success("Answer updated!")

            if st.button("🗑️ Delete", key=f"delete_{i}"):
                qa_data.pop(i)
                save_qa_log(qa_data)
                st.success("Deleted. Please refresh the page manually to see the update.")

    st.markdown("---")
    st.subheader("➕ Add Custom Question & Answer")
    new_name = st.text_input("👧 Child's Name", key="add_name")
    new_q = st.text_input("❓ New Question", key="add_q")
    new_a = st.text_area("💬 New Answer", key="add_a")

    if st.button("➕ Add to KB"):
        if new_q and new_a:
            new_entry = {
                "name": new_name,
                "question": new_q,
                "answer": new_a,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            qa_data.append(new_entry)
            save_qa_log(qa_data)
            st.success("Saved new Q&A to log! Please refresh to see it in the dashboard.")
        else:
            st.warning("Please enter both question and answer.")
