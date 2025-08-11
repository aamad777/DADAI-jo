import streamlit as st
import json
import os
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI

# project modules
from drawing import generate_drawing_with_stability
from sound import play_animal_sound
from dashboard import render_dashboard_tab
from learn import render_learning_book_tab
from kid_feedback import send_email_to_dad
from quiz_game import get_quiz_question
from quiz_sounds import play_correct_sound, play_wrong_sound, play_win_sound
from quiz_scoreboard import log_score, show_scoreboard

# NEW: drawing canvas + Gemini helpers
from streamlit_drawable_canvas import st_canvas
from gemini_ai import classify_sketch, fetch_animal_photo, ask_gemini

load_dotenv()

# Optional model keys
openai_api_key = os.getenv("OPENAI_API_KEY", "")
client = OpenAI(api_key=openai_api_key) if openai_api_key else None

# --------- EMBEDDED TTS (no external voice_io module needed) ----------
from io import BytesIO
def tts_gtts_bytes(text: str, lang: str = "en", slow: bool = False) -> bytes:
    """Generate speech (MP3 bytes) from text using gTTS."""
    if not text:
        return b""
    try:
        from gtts import gTTS  # requires: pip install gTTS
    except ImportError as e:
        raise ImportError("gTTS is not installed. Run: pip install gTTS") from e

    mp3_fp = BytesIO()
    tts = gTTS(text=text, lang=lang, slow=slow)
    tts.write_to_fp(mp3_fp)
    return mp3_fp.getvalue()
# ---------------------------------------------------------------------

# Page config
st.set_page_config(page_title="Ask Dad AI", layout="wide")

# Sidebar navigation
st.sidebar.title("📚Dad AI Navigation")
tab = st.sidebar.radio("Choose a tab:", [
    "💬 Ask Dad AI",
    "🐾 Animal Fun",
    "🛠️ Dad's Dashboard",
    "📚 Learning Book",
    "🧠 Quiz Fun",
    "📊 Scoreboard",
    "🎨 Draw & Guess (Gemini)"  # NEW TAB
])

# ----- helpers -----
def load_answers():
    try:
        with open("answers.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return {}

def save_qa_log(name, question, answer):
    entry = {
        "name": name,
        "question": question,
        "answer": answer,
        "timestamp": datetime.now().isoformat()
    }
    try:
        if os.path.exists("qa_log.json"):
            with open("qa_log.json", "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            data = []
    except:
        data = []
    data.append(entry)
    with open("qa_log.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def ask_predefined_or_model(question):
    # 1) predefined answers
    answers = load_answers()
    for k, v in answers.items():
        if k.lower() in question.lower():
            return v

    # 2) Gemini fallback (preferred)
    try:
        return ask_gemini(question)
    except Exception as gem_e:
        # 3) OpenAI fallback (optional)
        if client:
            try:
                resp = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are a friendly assistant for kids. Keep answers simple, kind, and short."},
                        {"role": "user", "content": question},
                    ],
                    temperature=0.4,
                    max_tokens=120
                )
                return resp.choices[0].message.content.strip()
            except Exception as openai_e:
                return f"Sorry, I couldn't answer right now: Gemini error: {gem_e}; OpenAI error: {openai_e}"
        return f"Sorry, I couldn't answer right now: {gem_e}"

# =============== TABS ===============

# 💬 Ask
if tab == "💬 Ask Dad AI":
    st.title("👨‍👧 Ask Dad AI")

    name = st.text_input("🙋 What's your name?", key="child_name")
    question = st.text_input("❓ What do you want to ask?", key="child_question")

    colA, colB = st.columns([1, 1])
    with colA:
        if st.button("✨ Get Answer"):
            kid = (name or "Kid").strip()
            if not question.strip():
                st.info("Please type a question.")
            else:
                answer = ask_predefined_or_model(question.strip())
                # Save to state so we can show feedback + email on every rerun
                st.session_state["last_kid"] = kid
                st.session_state["last_question"] = question.strip()
                st.session_state["last_answer"] = answer
                # clear last email status when new answer arrives
                st.session_state.pop("email_status", None)

    with colB:
        if st.button("🔊 Read Aloud"):
            answer = st.session_state.get("last_answer", "")
            if answer:
                try:
                    audio_bytes = tts_gtts_bytes(answer, lang="en", slow=False)
                    st.audio(audio_bytes, format="audio/mp3")
                except Exception as e:
                    st.error(f"TTS error: {e}")
            else:
                st.info("No answer to read yet. Ask a question first!")

    # --- Always show the result + feedback when we have an answer saved ---
    if st.session_state.get("last_answer"):
        kid = st.session_state.get("last_kid", "Kid")
        answer = st.session_state["last_answer"]
        question_saved = st.session_state.get("last_question", "")
        st.success(f"**Answer for {kid}:** {answer}")

        st.markdown("#### Did you understand the answer?")
        c_yes, c_no = st.columns(2)

        with c_yes:
            if st.button("👍 I understand", key="btn_understand_yes"):
                st.balloons()
                st.info("Awesome! You can try the Quiz Fun tab to practice 🍎")

        with c_no:
            if st.button("✉️ Email Dad (I don't understand)", key="btn_understand_no"):
                with st.spinner("Sending email to Dad..."):
                    ok, msg = send_email_to_dad(
                        child_name=kid,
                        question=question_saved,
                        answer=answer,
                    )
                # store + show status
                st.session_state["email_status"] = (ok, msg)

        # Show status if we have one from this or previous click
        if "email_status" in st.session_state:
            ok, msg = st.session_state["email_status"]
            if ok:
                st.success("Email sent to Dad ✅")
                st.caption(msg)
            else:
                st.error("Couldn't send email ❌")
                st.code(msg, language="text")

# 🐾 Animal Fun
elif tab == "🐾 Animal Fun":
    st.title("🐾 Animal Fun")
    st.write("Type an animal name to play a sound, or generate a fun drawing!")

    col1, col2 = st.columns(2)
    with col1:
        animal = st.text_input("Animal name (e.g., cat, dog, lion)")
        if st.button("🔊 Play Animal Sound"):
            if animal.strip():
                play_animal_sound(animal.strip().lower())
            else:
                st.info("Please enter an animal name.")

    with col2:
        prompt = st.text_input("Describe a drawing you want (e.g., 'cute baby lion with a crown')")
        if st.button("🎨 Generate Cute Drawing (Stability)"):
            img_bytes = generate_drawing_with_stability(prompt)
            if img_bytes:
                st.image(img_bytes, caption="Generated Art", use_container_width=True)
            else:
                st.warning("Couldn't generate drawing (check STABILITY_API_KEY in your .env).")

# 🛠️ Dad AI's Dashboard
elif tab == "🛠️ Dad's Dashboard":
    st.title("🛠️ Dad AI's Dashboard")

    # Optional quick test for email config
    if st.button("📧 Send Test Email to Dad"):
        with st.spinner("Sending test email..."):
            ok, msg = send_email_to_dad("Test Kid", "This is a test email.", "This is a test answer.")
        if ok:
            st.success("Test email sent ✅")
            st.caption(msg)
        else:
            st.error("Test email failed ❌")
            st.code(msg, language="text")

    render_dashboard_tab()

# 📚 Learning Book
elif tab == "📚 Learning Book":
    st.title("📚 Learning Book")
    render_learning_book_tab()

# 🧠 Quiz Fun
elif tab == "🧠 Quiz Fun":
    st.title("🧠 Quiz Fun")
    st.write("Answer fun questions and get stars!")
    name = st.text_input("Your name for the scoreboard:", key="quiz_name")
    if "quiz_started" not in st.session_state:
        st.session_state.quiz_started = False
    if "quiz_score" not in st.session_state:
        st.session_state.quiz_score = 0
    if "quiz_q_index" not in st.session_state:
        st.session_state.quiz_q_index = 0

    if not st.session_state.quiz_started:
        if st.button("▶️ Start Quiz"):
            st.session_state.quiz_started = True
            st.rerun()
    else:
        q = get_quiz_question(st.session_state.quiz_q_index)
        if q is None:
            st.subheader(f"🏁 Finished! Your score: {st.session_state.quiz_score}/5")
            stars = "⭐" * st.session_state.quiz_score + "☆" * (5 - st.session_state.quiz_score)
            st.markdown(f"### {stars}")
            st.image("https://media.giphy.com/media/l0MYt5jPR6QX5pnqM/giphy.gif", width=300)
            play_win_sound()
            if name.strip():
                log_score(name.strip(), st.session_state.quiz_score)

            if st.button("🔁 Play Again"):
                for key in list(st.session_state.keys()):
                    if key.startswith("quiz_"):
                        del st.session_state[key]
                st.rerun()
        else:
            st.subheader(q["question"])
            choice = st.radio("Pick one:", q["choices"], key=f"quiz_choice_{st.session_state.quiz_q_index}")
            if st.button("✅ Submit"):
                if choice == q["answer"]:
                    st.success("Correct! 🎉")
                    play_correct_sound()
                    st.session_state.quiz_score += 1
                else:
                    st.error(f"Oops! Correct answer is: {q['answer']}")
                    play_wrong_sound()
                st.session_state.quiz_q_index += 1
                st.rerun()

# 📊 Scoreboard
elif tab == "📊 Scoreboard":
    st.title("📊 Quiz Scoreboard")
    show_scoreboard()

# 🎨 Draw & Guess (Gemini)
elif tab == "🎨 Draw & Guess (Gemini)":
    st.title("🎨 Draw & Guess (Gemini)")
    st.write("Draw any animal and let AI guess it! Then see a real photo.")

    stroke_w = st.slider("Pen size", 4, 30, 12)
    bg = st.color_picker("Background", "#FFFFFF")
    st.write("Tip: use black pen on white background for best results.")

    canvas_result = st_canvas(
        fill_color="rgba(0, 0, 0, 0)",
        stroke_width=stroke_w,
        stroke_color="#000000",
        background_color=bg,
        update_streamlit=True,
        height=300,
        width=300,
        drawing_mode="freedraw",
        key="canvas",
    )

    col1, col2 = st.columns(2)
    guess = None
    if col1.button("🤖 Guess with Gemini"):
        if canvas_result.image_data is not None:
            from PIL import Image
            import numpy as np
            import io

            # Convert canvas (RGBA numpy array) to PNG bytes
            img = Image.fromarray(canvas_result.image_data.astype("uint8"), "RGBA")
            # Make it high-contrast B/W to help the model understand a kid’s sketch
            img = img.convert("L").point(lambda x: 0 if x < 250 else 255).convert("RGB")
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            png_bytes = buf.getvalue()

            with st.spinner("Asking Gemini..."):
                try:
                    guess = classify_sketch(png_bytes)
                except Exception as e:
                    st.error(f"Gemini error: {e}")
                    guess = None
        else:
            st.info("Please draw something first.")

    if guess:
        st.success(f"I think it's a **{guess.get('animal','unknown')}** (certainty {guess.get('certainty',0):.2f})")
        if guess.get("alternatives"):
            st.caption("Other ideas: " + ", ".join(guess["alternatives"]))

        if col2.button("📷 Show real photo"):
            with st.spinner("Finding a photo..."):
                url = fetch_animal_photo(guess.get("animal",""))
            if url:
                st.image(url, caption=f"Real photo of a {guess.get('animal','')}", use_container_width=True)
            else:
                st.warning("Couldn't find a photo right now. Try another animal or check your internet.")
