# app.py — Ask DAD AI (Classic + Neon UI + Name Fireworks + Compliment Cards + Magic Button + Explain 3 Ways)
import os, json, threading, queue, random, html
from datetime import datetime
from io import BytesIO

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

# ---- Optional mic widget (graceful fallback if not installed) ----
try:
    from audio_recorder_streamlit import audio_recorder  # pip install audio-recorder-streamlit
    HAS_AUDIO_RECORDER = True
except Exception:
    HAS_AUDIO_RECORDER = False

# ---- Project modules already in your repo ----
from drawing import generate_drawing_with_stability
from sound import play_animal_sound
from dashboard import render_dashboard_tab
from learn import render_learning_book_tab
from kid_feedback import send_email_to_dad
from quiz_game import get_quiz_question
from quiz_sounds import play_correct_sound, play_wrong_sound, play_win_sound
from quiz_scoreboard import log_score, show_scoreboard
from streamlit_drawable_canvas import st_canvas
from gemini_ai import classify_sketch, fetch_animal_photo, ask_gemini

# ---------------------------------------------------------------
load_dotenv()

# Optional OpenAI (fallback + Whisper STT)
openai_api_key = os.getenv("OPENAI_API_KEY", "")
client = OpenAI(api_key=openai_api_key) if openai_api_key else None

# ===== TTS (gTTS) =============================================================
def tts_gtts_bytes(text: str, lang: str = "en", slow: bool = False) -> bytes:
    if not text:
        return b""
    try:
        from gtts import gTTS
    except ImportError as e:
        raise ImportError("gTTS is not installed. Run: pip install gTTS") from e
    mp3_fp = BytesIO()
    gTTS(text=text, lang=lang, slow=slow).write_to_fp(mp3_fp)
    return mp3_fp.getvalue()

# ===== STT (Whisper if available; else Google with timeout) ===================
STT_TIMEOUT_SECS = 10

def _google_stt_worker(audio_bytes: bytes, out_q: "queue.Queue[tuple[str|None, str|None]]"):
    try:
        import speech_recognition as sr
        r = sr.Recognizer()
        r.dynamic_energy_threshold = True
        r.energy_threshold = 300
        with sr.AudioFile(BytesIO(audio_bytes)) as source:
            audio = r.record(source)
        text = r.recognize_google(audio)
        out_q.put((text, None))
    except Exception as e:
        out_q.put((None, f"STT failed: {e}"))

def transcribe_audio(audio_bytes: bytes):
    # Whisper first
    if os.getenv("OPENAI_API_KEY"):
        try:
            import tempfile
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            tmp.write(audio_bytes); tmp.flush(); tmp.close()
            client_local = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            with open(tmp.name, "rb") as f:
                resp = client_local.audio.transcriptions.create(model="whisper-1", file=f)
            text = (resp.text or "").strip()
            if text:
                return text, None
        except Exception:
            pass
    # Google in a thread with timeout
    out_q: "queue.Queue[tuple[str|None, str|None]]" = queue.Queue(maxsize=1)
    t = threading.Thread(target=_google_stt_worker, args=(audio_bytes, out_q), daemon=True)
    t.start(); t.join(STT_TIMEOUT_SECS)
    if t.is_alive():
        return None, f"STT timed out after {STT_TIMEOUT_SECS}s. Try again or type your question."
    try:
        text, err = out_q.get_nowait()
        return text, err
    except queue.Empty:
        return None, "STT failed unexpectedly (no result)."

# ===== Styling & Animations ===================================================
st.set_page_config(page_title="Ask DAD AI", layout="wide")

# Global colorful button style for Classic mode + floating emoji css for compliments
st.markdown("""
<style>
.kids-ui .stButton>button, .kids-ui [data-testid="stButton"]>button {
  font-size: 22px !important; font-weight: 800 !important; min-height: 56px !important;
  border-radius: 16px !important; border: none !important; color: #0f172a !important;
  box-shadow: 0 8px 16px rgba(0,0,0,0.08) !important; margin: 4px !important;
  background: linear-gradient(135deg,#fde68a,#fda4af 35%,#c4b5fd 70%,#93c5fd) !important;
  background-size: 400% 400%; animation: kidsRainbow 14s ease infinite;
  transition: transform .08s ease-in-out;
}
.kids-ui .stButton>button:hover { transform: scale(1.02); }
@keyframes kidsRainbow { 0%{background-position:0% 50%} 50%{background-position:100% 50%} 100%{background-position:0% 50%} }
.kids-ui [data-testid="stHorizontalBlock"] { gap: 6px !important; }
.kids-ui [data-testid="column"] { padding-left: 2px !important; padding-right: 2px !important; }

/* Name bubbles */
.name-bubbles { display:flex; flex-wrap:wrap; align-items:flex-end; gap:6px; }
.bubble {
  display:inline-block; padding:.26rem .52rem; border-radius:12px;
  color:#0b1324; font-weight:900; letter-spacing:.5px; line-height:1;
  background: radial-gradient(circle at 30% 20%, var(--c1), var(--c2));
  box-shadow: 0 6px 14px rgba(0,0,0,0.12);
  transform: translateY(12px) scale(.9); opacity:0;
  animation: popIn .5s ease forwards; animation-delay: var(--d);
  border: 2px solid rgba(255,255,255,.5);
}
@keyframes popIn { to { transform: translateY(0) scale(1); opacity:1; } }

.wave {
  font-size: 42px; font-weight: 900; letter-spacing: 1px; margin: 6px 0 4px 0;
  background: linear-gradient(90deg,#22c55e,#06b6d4,#a78bfa,#f97316);
  -webkit-background-clip: text; background-clip: text; color: transparent;
  animation: hue 6s linear infinite;
}
@keyframes hue { 0%{filter:hue-rotate(0deg)} 100%{filter:hue-rotate(360deg)} }

/* Compliment chips + floating emoji */
.compliment-row { display:flex; flex-wrap:wrap; gap:8px; margin: 8px 0 2px 0; }
.comp-chip {
  display:inline-flex; align-items:center; gap:8px; padding:10px 12px; border-radius:14px;
  background:linear-gradient(135deg,#e9d5ff,#bfdbfe); color:#0b1324; font-weight:900;
  box-shadow:0 8px 16px rgba(0,0,0,.08); border:1.5px solid rgba(255,255,255,.6);
}
.float-emoji { position:relative; display:inline-block; }
.float-emoji:before {
  content: "🎉";
  position:absolute; left:-6px; top:-10px; opacity:.85;
  animation: floatUp 1.6s ease-in-out infinite;
}
@keyframes floatUp {
  0%{ transform: translateY(4px); opacity:.8}
  50%{ transform: translateY(-4px); opacity:1}
  100%{ transform: translateY(4px); opacity:.8}
}
</style>
""", unsafe_allow_html=True)

# ===== Sidebar ================================================================
st.sidebar.title("📚 DAD AI Navigation")
tab = st.sidebar.radio("Choose a tab:", [
    "💬 Ask DAD AI",
    "🐾 Animal Fun",
    "🛠️ Dad's Dashboard",
    "📚 Learning Book",
    "🧠 Quiz Fun",
    "📊 Scoreboard",
    "🎨 Draw & Guess (Gemini)"
])

# Toggle UI style
ui_style = st.sidebar.selectbox("🎨 UI style", ["Classic", "Neon"], index=0)

# ===== Data helpers ===========================================================
def load_answers():
    try:
        with open("answers.json","r",encoding="utf-8") as f:
            return json.load(f)
    except:
        return {}

def save_qa_log(name, question, answer):
    entry = {"name":name,"question":question,"answer":answer,"timestamp":datetime.now().isoformat()}
    try:
        if os.path.exists("qa_log.json"):
            with open("qa_log.json","r",encoding="utf-8") as f:
                data = json.load(f)
        else:
            data = []
    except:
        data = []
    data.append(entry)
    with open("qa_log.json","w",encoding="utf-8") as f:
        json.dump(data,f,ensure_ascii=False,indent=2)

# ===== Categories & compliments ==============================================
CATEGORIES = {
    "Math": {"emoji": "➗", "colors": ("#fde68a", "#fca5a5"),
             "ideas": ["What is zero?", "What is 7 + 3?", "Why are triangles special?"]},
    "Science": {"emoji": "🔬", "colors": ("#bbf7d0", "#93c5fd"),
                "ideas": ["Why is the sky blue?", "What is gravity?", "How do plants drink water?"]},
    "Space": {"emoji": "🚀", "colors": ("#c7d2fe", "#93c5fd"),
              "ideas": ["What is a black hole?", "Why do stars twinkle?", "How big is the Sun?"]},
    "History": {"emoji": "🏛️", "colors": ("#fef3c7", "#fdba74"),
                "ideas": ["Who built the pyramids?", "Who was the first pilot?", "What is a castle?"]},
    "Animals": {"emoji": "🐼", "colors": ("#86efac", "#a7f3d0"),
                "ideas": ["Why do cats purr?", "How do bees make honey?", "Do elephants swim?"]},
    "Geography": {"emoji": "🗺️", "colors": ("#bae6fd", "#93c5fd"),
                  "ideas": ["Where does rain come from?", "What is a volcano?", "What is a desert?"]},
    "Art": {"emoji": "🎨", "colors": ("#fbcfe8", "#fda4af"),
            "ideas": ["What are primary colors?", "What is a portrait?", "How do you mix green?"]},
    "Sports": {"emoji": "⚽", "colors": ("#d1fae5", "#a7f3d0"),
               "ideas": ["How big is a soccer field?", "What is offside?", "Why do we warm up?"]},
}

# Three rotating compliments per age (we'll cycle an index)
AGE_COMPLIMENTS_3 = {
    1: ["🎈 Tiny explorer mode unlocked!", "🧩 Look at you discovering sounds!", "🌟 Little star on the move!"],
    2: ["🎈 Shape detective in action!", "🎶 Rhythm captain with claps!", "🧸 Cozy learner, big smiles!"],
    3: ["🎈 Super 3! Questions = magic keys!", "🔍 Curiosity booster activated!", "🚀 Learning rocket ready!"],
    4: ["🎈 Fantastic 4! Brain power zoom!", "🧠 Idea builder unlocked!", "🎨 Color master in training!"],
    5: ["🎈 High‑five 5! Mystery solver!", "🦸 Brave thinker level 5!", "📚 Word wizard warming up!"],
    6: ["🎈 Super six! Science ninja!", "⚙️ Tinkerer with bright ideas!", "🔬 Lab legend loading!"],
    7: ["🎈 Lucky 7! Space captain energy!", "🛰️ Orbiting awesome ideas!", "🌌 Galaxy of questions!"],
    8: ["🎈 Great 8! Math hero!", "🧩 Pattern pro unlocked!", "📏 Ruler of riddles!"],
    9: ["🎈 Brilliant 9! Ideas blasting off!", "💡 Spark storm incoming!", "🏆 Champion of ‘why’!"],
    10:["🎈 Terrific 10! Double‑awesome!", "🔥 Power‑up: expert learner!", "🎓 Junior scholar online!"],
}

# ===== Model wrapper: add category + age context ==============================
def ask_with_context(question: str, category: str | None, age: int | None) -> str:
    answers = load_answers()
    for k, v in answers.items():
        if k.lower() in question.lower():
            return v
    topic = category or "General"
    age_text = f"{age}" if age else "kid"
    instruction = (
        f"Please answer like a kind teacher for a child age {age_text}. "
        f"Topic: {topic}. Keep it short, clear, and fun. Use simple words."
    )
    # Try Gemini first
    try:
        return ask_gemini(f"{instruction}\nQuestion: {question}")
    except Exception:
        pass
    # Fallback OpenAI
    if client:
        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role":"system","content":instruction},
                          {"role":"user","content":question}],
                temperature=0.4, max_tokens=180
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:
            return f"Sorry, I couldn't answer right now: {e}"
    return "Sorry, I couldn't answer right now."

# ===== Audio input (mic if available, else WAV upload) ========================
def audio_input_ui():
    if HAS_AUDIO_RECORDER:
        st.caption("🎙️ Record your question")
        try:
            audio_bytes = audio_recorder(pause_threshold=1.0, sample_rate=16000, text="Tap to record / stop")
        except Exception:
            audio_bytes = None
        return audio_bytes, "recorder"
    else:
        st.caption("📁 Upload a short WAV clip (mic not available on this server)")
        file = st.file_uploader("Choose a .wav file", type=["wav"], accept_multiple_files=False, label_visibility="collapsed")
        if file:
            return file.read(), "upload"
        return None, "upload"

# ===== Name bubbles (animated) ================================================
BUBBLE_COLORS = [
    ("#a7f3d0", "#86efac"),
    ("#93c5fd", "#bfdbfe"),
    ("#fbcfe8", "#fda4af"),
    ("#fde68a", "#fca5a5"),
    ("#c7d2fe", "#93c5fd"),
    ("#fdba74", "#fef3c7"),
]
def bubble_name_html(name: str) -> str:
    if not name.strip():
        return "<div class='name-bubbles'> </div>"
    spans = []
    for i, ch in enumerate(name.strip()):
        c1, c2 = random.choice(BUBBLE_COLORS)
        delay = f"{0.05*i:.2f}s"
        safe = html.escape(ch.upper())
        spans.append(f"<span class='bubble' style='--c1:{c1};--c2:{c2};--d:{delay}'>{safe}</span>")
    return "<div class='name-bubbles'>" + "".join(spans) + "</div>"

# ===== Simple onboarding: name -> age -> ask ==================================

def name_step():
    st.markdown("<div class='kids-ui'>", unsafe_allow_html=True)
    st.subheader("🧩 What's your name?")

    # --- Name input WITHOUT a widget key; we manage it manually to avoid key mutation errors
    current_name = st.session_state.get("kid_name", "")
    typed = st.text_input("Type your name here:", value=current_name)
    if typed != current_name:
        st.session_state["kid_name"] = typed

    name = (st.session_state.get("kid_name","") or "").strip()

    st.caption("Preview")
    st.markdown(bubble_name_html(name), unsafe_allow_html=True)

    cols = st.columns([1,1,2])

    # Random name: set state BEFORE rendering next run (no widget key mutation)
    if cols[1].button("🎲 Random name"):
        demo = random.choice(["Maya","Omar","Lina","Adam","Sara","Ziad"])
        st.session_state["kid_name"] = demo
        st.rerun()

    if cols[0].button("👋 I'm ready!"):
        final_name = name or "Kid"
        st.session_state["child_name"] = final_name
        st.balloons()
        try:
            st.session_state["name_greeting_bytes"] = tts_gtts_bytes(f"Hi, {final_name}!", lang="en")
            st.session_state["play_name_greeting"] = True
        except Exception:
            st.session_state["name_greeting_bytes"] = b""
            st.session_state["play_name_greeting"] = False
        st.session_state["onboarding_step"] = "age"
        st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

def age_step():
    st.markdown("<div class='kids-ui'>", unsafe_allow_html=True)
    st.subheader("🎂 How old are you?")
    st.caption("Tap one")
    row = st.columns(10)
    picked = None
    for i, n in enumerate(range(1,11)):
        with row[i]:
            if st.button(str(n), key=f"age_{n}"):
                picked = n
    if picked is not None:
        st.session_state["kid_age"] = picked
        st.session_state["age_compliments_list"] = AGE_COMPLIMENTS_3.get(picked, ["🎈 Awesome age!"])
        st.session_state["age_comp_index"] = 0
        st.session_state["age_celebrate_msg"] = "age_ready"
        st.session_state["onboarding_step"] = "ask"
        st.rerun()

    name = st.session_state.get("child_name","Kid")
    st.markdown(f"<div class='wave'>Hi, {html.escape(name)}!</div>", unsafe_allow_html=True)
    st.markdown(bubble_name_html(name), unsafe_allow_html=True)

    # Auto-play name greeting once upon arriving to age step
    if st.session_state.pop("play_name_greeting", False):
        audio_bytes = st.session_state.pop("name_greeting_bytes", b"")
        if audio_bytes:
            st.audio(audio_bytes, format="audio/mp3")
    st.markdown("</div>", unsafe_allow_html=True)

def render_category_picker():
    st.markdown("#### 🎒 Pick a topic")
    keys = list(CATEGORIES.keys())
    cols_per_row = 4
    for start in range(0, len(keys), cols_per_row):
        row_keys = keys[start:start+cols_per_row]
        cols = st.columns(len(row_keys))
        for i, k in enumerate(row_keys):
            cfg = CATEGORIES[k]
            with cols[i]:
                st.markdown(
                    f"<div style=\"background:linear-gradient(135deg,{cfg['colors'][0]},{cfg['colors'][1]});"
                    f"padding:14px;border-radius:16px;color:#0b1324;font-weight:900;"
                    f"box-shadow:0 10px 22px rgba(0,0,0,.08);text-align:center\">"
                    f"<div style='font-size:36px'>{cfg['emoji']}</div>"
                    f"<div style='font-size:18px;margin-top:4px'>{k}</div>"
                    f"</div>",
                    unsafe_allow_html=True
                )
                if st.button(f"Choose {k}", key=f"choose_{k}", use_container_width=True):
                    st.session_state["topic_category"] = k
                    st.rerun()

def render_idea_chips(category: str):
    ideas = CATEGORIES.get(category, {}).get("ideas", [])
    if not ideas: return
    st.markdown("##### Try one of these:")
    chip_cols = st.columns(min(6, len(ideas)))
    for i, idea in enumerate(ideas):
        with chip_cols[i % len(chip_cols)]:
            if st.button(idea, key=f"idea_{category}_{i}"):
                st.session_state["child_question"] = idea
                st.rerun()

def _explain_three_ways(base_q: str, base_a: str, age: int | None, category: str | None):
    """
    Generate and cache Picture / Story / Steps variations for the current answer.
    Stored in st.session_state['explain3'] = {'picture': str, 'story': str, 'steps': str}
    """
    if "explain3" in st.session_state and st.session_state.get("explain3_q") == base_q:
        return st.session_state["explain3"]

    age_text = f"{age}" if age else "kid"
    topic = category or "General"

    def _gen(prompt: str) -> str:
        # Prefer Gemini
        try:
            return ask_gemini(prompt).strip()
        except Exception:
            pass
        # Fallback OpenAI
        if client:
            try:
                resp = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role":"user","content":prompt}],
                    temperature=0.5, max_tokens=220
                )
                return (resp.choices[0].message.content or "").strip()
            except Exception:
                return ""
        return ""

    picture = _gen(
        f"Explain with a visual description for a child age {age_text} about this topic ({topic}). "
        f"Describe a simple picture (text-only, no image) that would help them understand this answer: {base_a}. "
        f"Keep it to 2-3 cheerful sentences."
    )
    story = _gen(
        f"Tell a short, cute story (<=80 words) for a child age {age_text} that teaches the idea in this answer: {base_a}. "
        f"Use simple words and a friendly tone."
    )
    steps = _gen(
        f"Explain in 3–5 simple steps for a child age {age_text}, from easiest to a little harder, "
        f"to understand the idea behind this answer: {base_a}. Keep each step very short."
    )

    st.session_state["explain3"] = {"picture": picture, "story": story, "steps": steps}
    st.session_state["explain3_q"] = base_q
    return st.session_state["explain3"]

def ask_step():
    st.markdown("<div class='kids-ui'>", unsafe_allow_html=True)
    name = st.session_state.get("child_name","Kid")
    age = st.session_state.get("kid_age")
    category = st.session_state.get("topic_category")
    msg = st.session_state.pop("age_celebrate_msg", None)

    # Show rotating compliment chips when arriving from age step
    if msg == "age_ready":
        try: play_win_sound()
        except Exception: pass
        st.balloons()

        comps = st.session_state.get("age_compliments_list", [])
        idx = st.session_state.get("age_comp_index", 0)
        if comps:
            rot = comps[idx:] + comps[:idx]
            show = rot[:3] if len(rot) >= 3 else rot
            st.markdown("#### 🎉 You're awesome!")
            st.markdown("<div class='compliment-row'>", unsafe_allow_html=True)
            for c in show:
                st.markdown(f"<div class='comp-chip'><span class='float-emoji'></span>{html.escape(c)}</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
            if st.button("🔄 Another compliment"):
                st.session_state["age_comp_index"] = (idx + 1) % max(1, len(comps))
                st.session_state["age_celebrate_msg"] = "age_ready"
                st.rerun()

    st.markdown(f"### 👋 Hello, **{name}**" + (f" — age {age}" if age else ""))

    # Category picker + Magic Button
    if not category:
        render_category_picker()
    else:
        cfg = CATEGORIES[category]
        st.markdown(
            f"<div style='background:linear-gradient(135deg,{cfg['colors'][0]},{cfg['colors'][1]});"
            f"border-radius:14px;padding:10px 14px;color:#0b1324;font-weight:900;margin:6px 0;'>"
            f"{cfg['emoji']} Topic: {category}</div>", unsafe_allow_html=True
        )
        c1, c2, c3 = st.columns([1,1,1])
        with c1:
            if st.button("🔄 Change topic"):
                st.session_state.pop("topic_category", None); st.rerun()
        with c2:
            if st.button("🎯 More ideas"):
                random.shuffle(CATEGORIES[category]["ideas"]); st.rerun()
        with c3:
            if st.button("✨ Surprise me"):
                rand_cat = random.choice(list(CATEGORIES.keys()))
                st.session_state["topic_category"] = rand_cat
                idea = random.choice(CATEGORIES[rand_cat]["ideas"])
                st.session_state["prefill_child_question"] = idea
                st.rerun()

        render_idea_chips(category)

    default_q = st.session_state.pop("prefill_child_question", None)
    if default_q is not None:
        st.session_state.pop("child_question", None)
        question = st.text_input("❓ What do you want to ask?", value=default_q, key="ask_input")
    else:
        question = st.text_input("❓ What do you want to ask?", key="ask_input")

    audio_bytes, _ = audio_input_ui()
    if audio_bytes:
        st.audio(audio_bytes, format="audio/wav")
        with st.spinner("Transcribing…"):
            text, err = transcribe_audio(audio_bytes)
        if text:
            st.session_state['prefill_child_question'] = text
            st.success("✅ Added your question above."); st.rerun()
        else:
            st.error("🛑 Couldn't transcribe your audio.")
            if err: st.caption(err)

    c1, c2, c3 = st.columns([1,1,1])
    with c1:
        if st.button("✨ Get Answer", use_container_width=True):
            if not question.strip():
                st.info("Please type a question or use the mic/uploader.")
            else:
                answer = ask_with_context(question.strip(), st.session_state.get("topic_category"), age)
                st.session_state["last_answer"] = answer
                st.session_state["last_question"] = question.strip()
                st.session_state["just_answered"] = True
                save_qa_log(name, question.strip(), answer)
                st.rerun()
    with c2:
        if st.button("🔊 Read Aloud", use_container_width=True):
            answer = st.session_state.get("last_answer","")
            if answer:
                try:
                    st.audio(tts_gtts_bytes(answer, lang="en"), format="audio/mp3")
                except Exception as e:
                    st.error(f"TTS error: {e}")
            else:
                st.info("Ask something first!")
    with c3:
        if st.button("🔁 Ask another", use_container_width=True):
            st.session_state.pop("last_answer", None); st.session_state.pop("last_question", None); st.rerun()

    if st.session_state.get("last_answer"):
        if st.session_state.pop("just_answered", False):
            st.balloons()
            try: play_win_sound()
            except Exception: pass

        st.markdown("#### 🌟 Answer")
        st.success(st.session_state["last_answer"])

        # === Explain 3 Ways (tabs) ============================================
        st.markdown("#### Explain 3 Ways")
        tabs = st.tabs(["🖼 Picture", "📖 Story", "🪜 Steps"])
        e3 = _explain_three_ways(
            st.session_state.get("last_question",""),
            st.session_state["last_answer"],
            age,
            st.session_state.get("topic_category")
        )
        with tabs[0]:
            st.write(e3.get("picture",""))
            if st.button("🔁 Regenerate picture explain"):
                st.session_state.pop("explain3", None); st.rerun()
        with tabs[1]:
            st.write(e3.get("story",""))
            if st.button("🔁 Regenerate story"):
                st.session_state.pop("explain3", None); st.rerun()
        with tabs[2]:
            st.write(e3.get("steps",""))
            if st.button("🔁 Regenerate steps"):
                st.session_state.pop("explain3", None); st.rerun()

        # === Understanding + Email Dad ========================================
        st.markdown("#### Did you understand it?")
        y, n = st.columns(2)
        with y:
            if st.button("👍 I understand", key="understand_yes", use_container_width=True):
                st.info("Awesome! Want to try the Quiz tab too?")
        with n:
            if st.button("✉️ Email Dad (I don't understand)", key="understand_no", use_container_width=True):
                with st.spinner("Sending email to Dad..."):
                    ok, msg = send_email_to_dad(
                        child_name=name,
                        question=st.session_state.get("last_question",""),
                        answer=st.session_state["last_answer"]
                    )
                if ok:
                    st.success("Email sent to Dad ✅"); st.caption(msg)
                else:
                    st.error("Couldn't send email ❌"); st.code(msg, language="text")

    st.markdown("</div>", unsafe_allow_html=True)

# ===== NEON THEME =============================================================
NEON_CSS = """
<style>
:root{
  --neon:#3DF0A5; --neon2:#5BB9FF; --bg:#020617; --glass: rgba(15,23,42,.55);
  --txt:#E6F1FF; --dim:#90A4B4;
}
.neon-bg{ position:relative; background:radial-gradient(ellipse at top,#0b1b33 0%,#020617 60%);
  padding:8px; border-radius:24px; box-shadow:0 0 40px rgba(61,240,165,.08) inset;
}
.neon-grid:before{
  content:""; position:absolute; inset:-20px;
   background:
   radial-gradient(2px 2px at 20px 20px, rgba(93,185,255,.7), transparent 60%) 0 0/36px 36px,
   radial-gradient(2px 2px at 10px 10px, rgba(61,240,165,.6), transparent 60%) 0 0/30px 30px;
  filter: blur(.4px); opacity:.35; animation: drift 16s linear infinite;
}
@keyframes drift{ 0%{transform:translateY(0)} 100%{transform:translateY(-36px)} }

.neon-header{
  display:flex; align-items:center; gap:10px; padding:10px 16px; border-radius:16px;
  background:linear-gradient(90deg, rgba(61,240,165,.18), rgba(91,185,255,.18));
  border:2px solid rgba(91,185,255,.35);
  box-shadow:0 0 16px rgba(61,240,165,.25), inset 0 0 20px rgba(91,185,255,.12);
  color:var(--txt); font-weight:900; letter-spacing:.5px;
}
.neon-chip{font-size:14px;color:var(--dim);margin-left:auto}

.neon-window{
  margin-top:10px; border-radius:26px; padding:16px; min-height:360px;
   background:linear-gradient(180deg, rgba(10,19,39,.85), rgba(6,12,25,.85));
  border:2px solid rgba(61,240,165,.35);
  box-shadow:0 0 18px rgba(61,240,165,.25), inset 0 0 28px rgba(91,185,255,.14);
  color:var(--txt);
}
.msg{max-width:80%; padding:10px 14px; border-radius:14px; margin:8px 0; display:inline-block}
.msg.user{background:rgba(91,185,255,.18); border:1px solid rgba(91,185,255,.45)}
.msg.bot{background:rgba(61,240,165,.16); border:1px solid rgba(61,240,165,.45)}
.row{display:flex;gap:8px} .right{justify-content:flex-end}

.neon-input{
  margin-top:12px; display:flex; gap:8px; align-items:center;
  background:var(--glass); border:2px solid rgba(91,185,255,.35); border-radius:18px; padding:8px;
  box-shadow: inset 0 0 14px rgba(91,185,255,.10);
}
.neon-input input{ background:transparent!important; color:var(--txt)!important; }
.neon-send button{
  width:64px; height:44px; border-radius:12px; border:2px solid rgba(61,240,165,.55)!important;
  background:linear-gradient(180deg, rgba(61,240,165,.25), rgba(91,185,255,.20))!important;
  color:#0d1b2a!important; font-weight:900; box-shadow:0 0 18px rgba(61,240,165,.28);
}
.small{font-size:12px;color:var(--dim)}
</style>
"""

def render_neon_chat_ui():
    st.markdown(NEON_CSS, unsafe_allow_html=True)
    age  = st.session_state.get("kid_age")
    category = st.session_state.get("topic_category","General")

    if "neon_chat" not in st.session_state:
        st.session_state.neon_chat = []

    st.markdown('<div class="neon-bg neon-grid">', unsafe_allow_html=True)
    st.markdown(f'''
      <div class="neon-header">
        <span>🤖 Ask DAD AI</span>
        <span class="neon-chip">Topic: {category} • Age: {age or "?"}</span>
      </div>''', unsafe_allow_html=True)

    st.markdown('<div class="neon-window">', unsafe_allow_html=True)
    if not st.session_state.neon_chat:
        st.markdown('<div class="small">How can I help you?</div>', unsafe_allow_html=True)
    for role, text in st.session_state.neon_chat[-12:]:
        cls = "right" if role=="user" else ""
        who = "user" if role=="user" else "bot"
        st.markdown(f'<div class="row {cls}"><div class="msg {who}">{html.escape(text)}</div></div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    col_in, col_btn = st.columns([1,0.18])
    with col_in:
        q = st.text_input(" ", key="child_question", label_visibility="collapsed",
                          placeholder="Type your question…")
    with col_btn:
        send = st.button("➤", key="neon_send", use_container_width=True, help="Send")

    if send and q.strip():
        st.session_state.neon_chat.append(("user", q.strip()))
        try:
            ans = ask_with_context(q.strip(), st.session_state.get("topic_category"), age)
        except Exception:
            ans = "Sorry, couldn't answer right now."
        st.session_state.neon_chat.append(("bot", ans))
        st.session_state["last_answer"] = ans
        st.session_state["last_question"] = q.strip()
        st.session_state["child_question"] = ""
        st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

# ========================= MAIN TABS ==========================================
if tab == "💬 Ask DAD AI":
    st.title("👨‍👧 Ask DAD AI")
    step = st.session_state.get("onboarding_step", "name")
    if step == "name":
        name_step()
    elif step == "age":
        age_step()
    else:
        if "child_name" not in st.session_state:
            st.session_state["child_name"] = (st.session_state.get("kid_name","") or "Kid").strip()
        if ui_style == "Neon":
            render_neon_chat_ui()
        else:
            ask_step()

elif tab == "🐾 Animal Fun":
    st.title("🐾 Animal Fun"); st.write("Type an animal name to play a sound, or generate a fun drawing!")
    col1, col2 = st.columns(2)
    with col1:
        animal = st.text_input("Animal name (e.g., cat, dog, lion)")
        if st.button("🔊 Play Animal Sound"):
            if animal.strip(): play_animal_sound(animal.strip().lower())
            else: st.info("Please enter an animal name.")
    with col2:
        prompt = st.text_input("Describe a drawing you want (e.g., 'cute baby lion with a crown')")
        if st.button("🎨 Generate Cute Drawing (Stability)"):
            img_bytes = generate_drawing_with_stability(prompt)
            if img_bytes: st.image(img_bytes, caption="Generated Art", use_container_width=True)
            else: st.warning("Couldn't generate drawing (check STABILITY_API_KEY in your .env).")

elif tab == "🛠️ Dad's Dashboard":
    st.title("🛠️ Dad's Dashboard")
    if st.button("📧 Send Test Email to Dad"):
        with st.spinner("Sending test email..."):
            ok, msg = send_email_to_dad("Test Kid","This is a test email.","This is a test answer.")
        if ok: st.success("Test email sent ✅"); st.caption(msg)
        else: st.error("Test email failed ❌"); st.code(msg, language="text")
    render_dashboard_tab()

elif tab == "📚 Learning Book":
    st.title("📚 Learning Book"); render_learning_book_tab()

elif tab == "🧠 Quiz Fun":
    st.title("🧠 Quiz Fun"); st.write("Answer fun questions and get stars!")
    name = st.text_input("Your name for the scoreboard:", key="quiz_name")
    if "quiz_started" not in st.session_state: st.session_state.quiz_started = False
    if "quiz_score" not in st.session_state: st.session_state.quiz_score = 0
    if "quiz_q_index" not in st.session_state: st.session_state.quiz_q_index = 0
    if not st.session_state.quiz_started:
        if st.button("▶️ Start Quiz"):
            st.session_state.quiz_started = True; st.rerun()
    else:
        q = get_quiz_question(st.session_state.quiz_q_index)
        if q is None:
            st.subheader(f"🏁 Finished! Your score: {st.session_state.quiz_score}/5")
            stars = "⭐"*st.session_state.quiz_score + "☆"*(5 - st.session_state.quiz_score)
            st.markdown(f"### {stars}")
            st.image("https://media.giphy.com/media/l0MYt5jPR6QX5pnqM/giphy.gif", width=300)
            play_win_sound()
            if name.strip(): log_score(name.strip(), st.session_state.quiz_score)
            if st.button("🔁 Play Again"):
                for key in list(st.session_state.keys()):
                    if key.startswith("quiz_"): del st.session_state[key]
                st.rerun()
        else:
            st.subheader(q["question"])
            choice = st.radio("Pick one:", q["choices"], key=f"quiz_choice_{st.session_state.quiz_q_index}")
            if st.button("✅ Submit"):
                if choice == q["answer"]:
                    st.success("Correct! 🎉"); play_correct_sound(); st.session_state.quiz_score += 1
                else:
                    st.error(f"Oops! Correct answer is: {q['answer']}"); play_wrong_sound()
                st.session_state.quiz_q_index += 1; st.rerun()

elif tab == "📊 Scoreboard":
    st.title("📊 Quiz Scoreboard"); show_scoreboard()

elif tab == "🎨 Draw & Guess (Gemini)":
    st.title("🎨 Draw & Guess (Gemini)")
    stroke_w = st.slider("Pen size", 4, 30, 12); bg = st.color_picker("Background", "#FFFFFF")
    st.write("Tip: use black pen on white background for best results.")
    canvas_result = st_canvas(fill_color="rgba(0, 0, 0, 0)", stroke_width=stroke_w, stroke_color="#000000",
                              background_color=bg, update_streamlit=True, height=300, width=300,
                              drawing_mode="freedraw", key="canvas")
    col1, col2 = st.columns(2); guess = None
    if col1.button("🤖 Guess with Gemini"):
        if canvas_result.image_data is not None:
            from PIL import Image
            img = Image.fromarray(canvas_result.image_data.astype("uint8"), "RGBA")
            img = img.convert("L").point(lambda x: 0 if x < 250 else 255).convert("RGB")
            buf = BytesIO(); img.save(buf, format="PNG"); png_bytes = buf.getvalue()
            with st.spinner("Asking Gemini..."):
                try:
                    guess = classify_sketch(png_bytes)
                except Exception as e:
                    st.error(f"Gemini error: {e}"); guess = None
        else:
            st.info("Please draw something first.")
    if guess:
        st.success(f"I think it's a **{guess.get('animal','unknown')}** (certainty {guess.get('certainty',0):.2f})")
        if guess.get("alternatives"): st.caption("Other ideas: " + ", ".join(guess["alternatives"]))
        if col2.button("📷 Show real photo"):
            with st.spinner("Finding a photo..."):
                url = fetch_animal_photo(guess.get("animal",""))
            if url: st.image(url, caption=f"Real photo of a {guess.get('animal','')}", use_container_width=True)
            else: st.warning("Couldn't find a photo right now. Try another animal or check your internet.")
