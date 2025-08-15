# dashboard.py — minimal demo widgets
import streamlit as st
import json, os, datetime

def render_dashboard_tab():
    st.subheader("Family Log")
    st.caption("This is a tiny demo dashboard. Replace with your own metrics.")

    # Quick log view
    if os.path.exists("qa_log.json"):
        with open("qa_log.json","r",encoding="utf-8") as f:
            data = json.load(f)
        st.write(f"Total Q&A logs: {len(data)}")
        for row in data[-5:][::-1]:
            st.write(f"**{row['name']}** asked: _{row['question']}_ → {row['answer'][:80]}…")
    else:
        st.info("No Q&A log yet. Ask a question in the main tab!")
