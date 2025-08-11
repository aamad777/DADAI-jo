import streamlit as st
import os
import fitz  # PyMuPDF

BOOK_FILE = "learning_book.txt"

def render_learning_book_tab():
    st.title("📚 Upload a Learning Book")
    uploaded = st.file_uploader("Upload a .txt or .pdf file", type=["txt", "pdf"])

    if uploaded:
        if uploaded.name.endswith(".txt"):
            text = uploaded.read().decode("utf-8")
        elif uploaded.name.endswith(".pdf"):
            text = ""
            with fitz.open(stream=uploaded.read(), filetype="pdf") as doc:
                for page in doc:
                    text += page.get_text()
        else:
            st.error("Unsupported file type.")
            return

        with open(BOOK_FILE, "w", encoding="utf-8") as f:
            f.write(text)

        st.success("Book uploaded and saved!")
