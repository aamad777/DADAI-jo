# app.py — DAD AI (Arabic + English + Book Store + Name Meaning + Kid Hero)
import os, re, json, random, html, tempfile, shutil, io, threading, queue, base64
from io import BytesIO
from datetime import datetime
from pathlib import Path

import numpy as np
import streamlit as st
from dotenv import load_dotenv

# ====== Optional mic widget ======
try:
    from audio_recorder_streamlit import audio_recorder
    HAS_AUDIO_RECORDER = True
except Exception:
    HAS_AUDIO_RECORDER = False

# ====== AI Clients (Gemini + OpenAI) ======
import google.generativeai as genai
from openai import OpenAI

# ====== PDF & OCR ======
import fitz  # PyMuPDF
from pdfminer.high_level import extract_text as pdfminer_extract
try:
    import pytesseract
    from pdf2image import convert_from_bytes
    OCR_OK = True
except Exception:
    OCR_OK = False

# ====== Embeddings ======
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ====== Misc ======
from PIL import Image

# ------------------------------------------------------------------------------
# Load env
# ------------------------------------------------------------------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# ------------------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------------------
ROOT = Path(__file__).parent
BOOKS_DIR = ROOT / "books"       # persistent store for uploaded books
BOOKS_DIR.mkdir(exist_ok=True)

# ------------------------------------------------------------------------------
# Language helpers
# ------------------------------------------------------------------------------
LANGS = {"English": "en", "العربية": "ar"}
if "lang" not in st.session_state:
    st.session_state["lang"] = "en"

def get_lang():
    return st.session_state.get("lang", "en")

def is_ar():
    return get_lang().startswith("ar")

def tr(en, ar):
    return ar if is_ar() else en

def _lang_hint():
    return "Respond in Arabic (Modern Standard Arabic) with very simple words." if is_ar() \
           else "Respond in English with very simple words."

# ------------------------------------------------------------------------------
# TTS (gTTS)
# ------------------------------------------------------------------------------
def tts_gtts_bytes(text: str, lang: str = "en", slow: bool = False) -> bytes:
    if not text:
        return b""
    try:
        from gtts import gTTS
    except ImportError as e:
        raise ImportError("gTTS is not installed. Add gTTS to requirements.") from e
    mp3_fp = BytesIO()
    gTTS(text=text, lang=lang, slow=slow).write_to_fp(mp3_fp)
    return mp3_fp.getvalue()

# ------------------------------------------------------------------------------
# STT (Whisper first, then Google SR with timeout)
# ------------------------------------------------------------------------------
STT_TIMEOUT_SECS = 10

def _google_stt_worker(audio_bytes: bytes, out_q: "queue.Queue[tuple[str|None, str|None]]"):
    try:
        import speech_recognition as sr
        r = sr.Recognizer()
        r.dynamic_energy_threshold = True
        r.energy_threshold = 300
        with sr.AudioFile(BytesIO(audio_bytes)) as source:
            audio = r.record(source)
        lang_code = "ar-SA" if is_ar() else "en-US"
        text = r.recognize_google(audio, language=lang_code)
        out_q.put((text, None))
    except Exception as e:
        out_q.put((None, f"STT failed: {e}"))

def transcribe_audio(audio_bytes: bytes):
    # Whisper first
    if OPENAI_API_KEY:
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(audio_bytes)
                tmp.flush()
                tmp_path = tmp.name
            with open(tmp_path, "rb") as f:
                resp = client.audio.transcriptions.create(model="whisper-1", file=f)
            os.unlink(tmp_path)
            if resp and getattr(resp, "text", ""):
                return resp.text.strip(), None
        except Exception:
            pass
    # Google fallback (thread + timeout)
    out_q: "queue.Queue[tuple[str|None, str|None]]" = queue.Queue(maxsize=1)
    t = threading.Thread(target=_google_stt_worker, args=(audio_bytes, out_q), daemon=True)
    t.start(); t.join(STT_TIMEOUT_SECS)
    if t.is_alive():
        return None, tr(f"STT timed out after {STT_TIMEOUT_SECS}s.", f"انتهت مهلة تحويل الصوت بعد {STT_TIMEOUT_SECS}ث.")
    try:
        text, err = out_q.get_nowait()
        return text, err
    except queue.Empty:
        return None, tr("STT failed unexpectedly.", "تعذّر التحويل الصوتي بشكل غير متوقع.")

# ------------------------------------------------------------------------------
# Styles (kids buttons + colorful A/B/C + colorful number chips)
# ------------------------------------------------------------------------------
st.set_page_config(page_title="DAD AI", layout="wide")
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

.alpha-chip {
  display:inline-flex; align-items:center; justify-content:center;
  width:28px; height:28px; border-radius:50%;
  margin-right:8px; font-weight:900; color:#0b1324;
  background: linear-gradient(135deg,var(--g1),var(--g2));
  box-shadow:0 4px 10px rgba(0,0,0,.12);
}
.alpha-row { display:flex; align-items:flex-start; gap:8px; margin:6px 0; }
.number-chip {
  display:inline-flex; align-items:center; justify-content:center;
  width:32px; height:32px; border-radius:50%; margin: 4px; font-weight:900;
  color:#0b1324; background:linear-gradient(135deg,#fef3c7,#fdba74);
  box-shadow:0 4px 10px rgba(0,0,0,.12); border:2px solid rgba(255,255,255,.6);
}
</style>
""", unsafe_allow_html=True)

# RTL tweaks
if is_ar():
    st.markdown("""
    <style>
      html, body, [data-testid="stAppViewContainer"] * { direction: rtl; text-align: right; }
      .alpha-row { flex-direction: row-reverse; }
      .alpha-chip { margin-right:0; margin-left:8px; }
    </style>
    """, unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# Sidebar (language + tabs)
# ------------------------------------------------------------------------------
st.sidebar.title("📚 DAD AI")
lang_choice = st.sidebar.selectbox(tr("Language","اللغة"), list(LANGS.keys()),
                                   index=0 if get_lang()=="en" else 1)
st.session_state["lang"] = LANGS[lang_choice]

TABS = [
    tr("💬 Ask DAD AI","💬 اسأل بابا الذكي"),
    tr("📚 Learning Book","📚 كتاب التعلم"),
    tr("📚 Book Store (choose & manage)","📚 مخزن الكتب (اختيار وإدارة)"),
    tr("🐾 Animal Fun","🐾 مرح مع الحيوانات"),
    tr("🧠 Quiz Fun","🧠 مسابقة ممتعة"),
    tr("📊 Scoreboard","📊 النتائج"),
    tr("🎨 Draw & Guess","🎨 ارسم وخمّن"),
]
tab = st.sidebar.radio(tr("Choose a tab:","اختر صفحة:"), TABS, index=0)

# ------------------------------------------------------------------------------
# Helpers: name bubbles & colorful A/B/C steps
# ------------------------------------------------------------------------------
BUBBLE_COLORS = [
    ("#a7f3d0", "#86efac"), ("#93c5fd", "#bfdbfe"), ("#fbcfe8", "#fda4af"),
    ("#fde68a", "#fca5a5"), ("#c7d2fe", "#93c5fd"), ("#fdba74", "#fef3c7"),
]
ALPHA_COLORS = [
    ("#fde68a","#fca5a5"), ("#bbf7d0","#86efac"), ("#c7d2fe","#93c5fd"),
    ("#fbcfe8","#fda4af"), ("#bae6fd","#93c5fd"), ("#d1fae5","#a7f3d0"),
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

def render_alpha_steps(text_block: str):
    if not text_block.strip():
        return
    lines = [ln.strip() for ln in text_block.splitlines() if ln.strip()]
    cleaned = []
    for ln in lines:
        ln = re.sub(r"^[\-\*\d]+\s*[)\.．】）］]?\s*", "", ln).strip()
        cleaned.append(ln)
    for i, ln in enumerate(cleaned):
        g1, g2 = random.choice(ALPHA_COLORS)
        chip = f"<span class='alpha-chip' style='--g1:{g1};--g2:{g2}'>{chr(65+i)}</span>"
        st.markdown(f"<div class='alpha-row'>{chip}<div>{html.escape(ln)}</div></div>", unsafe_allow_html=True)

def number_chip(n: int) -> str:
    return f"<span class='number-chip'>{n}</span>"

# ------------------------------------------------------------------------------
# Name meaning + Kid hero
# ------------------------------------------------------------------------------
NAME_MEANINGS = {
    "en": {
        "adam": "Means 'earth'—a classic name.",
        "lina": "Means 'tender' or 'gentle'.",
        "maya": "Often linked to 'water' or 'magic'.",
        "omar": "Means 'long-lived' or 'prosperous'.",
        "sara": "Means 'joy' or 'princess'.",
        "ziad": "Means 'growth' or 'abundance'.",
    },
    "ar": {
        "آدم": "يعني «الأرض» — اسم عريق.",
        "لينا": "يعني «الرفيقة اللطيفة».",
        "مايا": "يرتبط بالماء أو السحر.",
        "عمر": "يعني «طويل العمر» أو «مزدهر».",
        "سارة": "يعني «الفرح» أو «الأميرة».",
        "زياد": "يعني «النماء» أو «الزيادة».",
    }
}
def _normalize_name(n: str) -> str:
    return (n or "").strip().lower()

def lookup_name_meaning_local(name: str) -> str | None:
    if is_ar():
        return NAME_MEANINGS["ar"].get(name.strip()) or None
    key = _normalize_name(name)
    return NAME_MEANINGS["en"].get(key)

def lookup_name_meaning_ai(name: str) -> str:
    prompt = (
        f"{_lang_hint()} Give a kind, super-short meaning or origin of the given first name. "
        f"One sentence only. Name: {name}"
    )
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        out = model.generate_content(prompt)
        txt = (getattr(out, "text", "").strip() or "").strip()
        if txt:
            return txt
    except Exception:
        pass
    if client:
        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role":"user","content":prompt}],
                temperature=0.2, max_tokens=48
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception:
            pass
    return tr("A lovely name!","اسم جميل!")

KID_HEROES = [
    {
        "name_en": "Nurse", "name_ar": "الممرّضة",
        "img": "https://upload.wikimedia.org/wikipedia/commons/thumb/6/6d/Cartoon_nurse.svg/512px-Cartoon_nurse.svg.png",
        "fact_en": "Heroes who care for people and give them medicine.",
        "fact_ar": "أبطال يعتنون بالناس ويقدّمون الدواء.",
    },
    {
        "name_en": "Firefighter", "name_ar": "رجُل الإطفاء",
        "img": "https://upload.wikimedia.org/wikipedia/commons/thumb/7/76/Firefighter_icon.svg/512px-Firefighter_icon.svg.png",
        "fact_en": "They put out fires and keep everyone safe.",
        "fact_ar": "يطفئون الحرائق ويحفظون سلامة الجميع.",
    },
    {
        "name_en": "Astronaut", "name_ar": "رائد الفضاء",
        "img": "https://upload.wikimedia.org/wikipedia/commons/thumb/5/58/Astronaut_simple.svg/512px-Astronaut_simple.svg.png",
        "fact_en": "They explore space and study the stars.",
        "fact_ar": "يستكشفون الفضاء ويدرسون النجوم.",
    },
    {
        "name_en": "Engineer", "name_ar": "المهندس",
        "img": "https://upload.wikimedia.org/wikipedia/commons/thumb/5/5e/Engineer_icon.svg/512px-Engineer_icon.svg.png",
        "fact_en": "They design and build useful things.",
        "fact_ar": "يصمّمون ويَبنون أشياء مفيدة.",
    },
]
def random_hero_card():
    h = random.choice(KID_HEROES)
    title = h["name_ar"] if is_ar() else h["name_en"]
    fact  = h["fact_ar"] if is_ar() else h["fact_en"]
    st.markdown(tr("**Today's little hero**","**بطلنا الصغير اليوم**"))
    cols = st.columns([1,2])
    with cols[0]:
        st.image(h["img"], use_container_width=True)
    with cols[1]:
        st.markdown(f"### {title}")
        st.caption(fact)

def render_name_extras(name: str):
    if not name.strip():
        return
    m = lookup_name_meaning_local(name)
    if not m:
        m = lookup_name_meaning_ai(name)
    st.info(tr(f"**Name meaning:** {m}", f"**معنى الاسم:** {m}"))
    random_hero_card()

# ------------------------------------------------------------------------------
# Gemini/OpenAI wrapper
# ------------------------------------------------------------------------------
def ask_gemini(prompt: str) -> str:
    model = genai.GenerativeModel("gemini-1.5-flash")
    out = model.generate_content(prompt)
    return (getattr(out, "text", "").strip() or "").strip()

def ask_simple(question: str, system: str = "") -> str:
    # Gemini first
    try:
        return ask_gemini((system + "\n\n" if system else "") + question)
    except Exception:
        pass
    # OpenAI fallback
    if client:
        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=([{"role":"system","content":system}] if system else []) + \
                         [{"role":"user","content":question}],
                temperature=0.4, max_tokens=220
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:
            return tr(f"Sorry, I couldn't answer: {e}", f"عذراً، لا أستطيع الإجابة: {e}")
    return tr("Sorry, I couldn't answer right now.","عذراً، لا أستطيع الإجابة الآن.")

# ------------------------------------------------------------------------------
# Embeddings model (cached)
# ------------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_embed_model():
    return SentenceTransformer("distiluse-base-multilingual-cased-v1")

# ------------------------------------------------------------------------------
# PDF extraction (text + OCR)
# ------------------------------------------------------------------------------
def extract_text_pymupdf(pdf_bytes: bytes) -> str:
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        parts = []
        for p in doc:
            t = p.get_text("text")
            if t and t.strip():
                parts.append(t)
        return "\n".join(parts)
    except Exception:
        return ""

def extract_text_pdfminer(pdf_bytes: bytes) -> str:
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(pdf_bytes); tmp.flush()
            txt = pdfminer_extract(tmp.name) or ""
        os.unlink(tmp.name)
        return txt
    except Exception:
        return ""

def extract_text_ocr(pdf_bytes: bytes) -> str:
    if not OCR_OK:
        return ""
    try:
        poppler_path = os.getenv("POPPLER_PATH")  # optional
        pages = convert_from_bytes(pdf_bytes, poppler_path=poppler_path) if poppler_path else convert_from_bytes(pdf_bytes)
        lang = "ara+eng" if is_ar() else "eng"
        pieces = []
        for img in pages[:40]:  # safety cap
            pieces.append(pytesseract.image_to_string(img, lang=lang))
        return "\n".join(pieces)
    except Exception:
        return ""

def extract_text_any(pdf_bytes: bytes) -> tuple[str, bool]:
    # return (text, used_ocr)
    txt = extract_text_pymupdf(pdf_bytes)
    if len(txt.strip()) >= 50:
        return txt, False
    txt = extract_text_pdfminer(pdf_bytes)
    if len(txt.strip()) >= 50:
        return txt, False
    txt = extract_text_ocr(pdf_bytes)
    return txt, True

# ------------------------------------------------------------------------------
# Book Store: chunking, indexing, search
# ------------------------------------------------------------------------------
def chunk_text(text: str, chunk_size=900, overlap=150):
    text = re.sub(r"\s+", " ", text).strip()
    chunks, i = [], 0
    while i < len(text):
        chunks.append(text[i:i+chunk_size])
        i += max(1, chunk_size - overlap)
    return chunks

def save_json(path: Path, data: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def load_json(path: Path, default=None):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default

def build_book_index(raw_text: str, title: str) -> dict:
    model = load_embed_model()
    chunks = chunk_text(raw_text)
    if not chunks:
        return {"title": title, "chunks": [], "embeddings_path": None}
    embs = model.encode(chunks, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)
    # store npy to disk
    book_id = re.sub(r"[^a-zA-Z0-9_-]+", "_", title) + "_" + str(int(datetime.utcnow().timestamp()))
    book_dir = BOOKS_DIR / book_id
    book_dir.mkdir(parents=True, exist_ok=True)
    npy_path = book_dir / "embeddings.npy"
    np.save(npy_path, embs)
    meta = {"title": title, "chunks": chunks, "embeddings_path": str(npy_path)}
    save_json(book_dir / "meta.json", meta)
    return {"book_id": book_id, **meta}

def list_books():
    books = []
    for p in BOOKS_DIR.iterdir():
        if p.is_dir() and (p / "meta.json").exists():
            meta = load_json(p / "meta.json", {})
            meta["book_id"] = p.name
            books.append(meta)
    books.sort(key=lambda x: x.get("title",""))
    return books

def delete_book(book_id: str):
    d = BOOKS_DIR / book_id
    if d.exists():
        shutil.rmtree(d, ignore_errors=True)

def load_book(book_id: str) -> dict | None:
    d = BOOKS_DIR / book_id
    if (d / "meta.json").exists():
        meta = load_json(d / "meta.json", {})
        meta["book_id"] = book_id
        return meta
    return None

def search_book(book_meta: dict, query: str, top_k=4) -> list[str]:
    if not book_meta or not book_meta.get("chunks"):
        return []
    npy = book_meta.get("embeddings_path")
    if not npy or not os.path.exists(npy):
        return []
    embs = np.load(npy)
    model = load_embed_model()
    q = model.encode([query], normalize_embeddings=True)
    sims = cosine_similarity(q, embs)[0]
    idxs = sims.argsort()[::-1][:top_k]
    return [book_meta["chunks"][i] for i in idxs]

# ------------------------------------------------------------------------------
# Ask with book-first fallback
# ------------------------------------------------------------------------------
def ask_with_book_first(question: str, age: int | None) -> str:
    # 1) if we have active book
    active_id = st.session_state.get("active_book_id")
    book_meta = load_book(active_id) if active_id else None
    if book_meta:
        ctx = "\n\n".join(search_book(book_meta, question, top_k=4))
        if ctx.strip():
            prompt = (
                f"Use the following book excerpts to answer the kid's question.\n\n"
                f"EXCERPTS:\n{ctx}\n\n"
                f"QUESTION: {question}\n\n"
                f"RULES: {_lang_hint()} Keep it short and kind. If unsure, say you aren't sure."
            )
            ans = ask_simple(prompt)
            if ans and not re.search(r"not sure|غير متأكد|لا أعلم", ans, re.I):
                return ans
    # 2) fallback to general AI
    system = f"You're a kind teacher for a child age {age or tr('kid','طفل')}. {_lang_hint()}"
    return ask_simple(question, system)

# ------------------------------------------------------------------------------
# Onboarding (name -> age) with number chips and meaning/hero
# ------------------------------------------------------------------------------
def name_step():
    st.markdown("<div class='kids-ui'>", unsafe_allow_html=True)
    st.subheader(tr("🧩 What's your name?","🧩 ما اسمك؟"))

    current_name = st.session_state.get("kid_name", "")
    typed = st.text_input(tr("Type your name here:","اكتب اسمك هنا:"), value=current_name, key="kid_name_input")
    if typed != current_name:
        st.session_state["kid_name"] = typed

    name = (st.session_state.get("kid_name","") or "").strip()
    st.caption(tr("Preview","معاينة"))
    st.markdown(bubble_name_html(name), unsafe_allow_html=True)

    if name:
        render_name_extras(name)

    cols = st.columns([1,1,2])
    if cols[1].button(tr("🎲 Random name","🎲 اسم عشوائي")):
        demo = random.choice(["Maya","Omar","Lina","Adam","Sara","Ziad"])
        st.session_state["kid_name"] = demo
        st.rerun()
    if cols[0].button(tr("👋 I'm ready!","👋 أنا جاهز!")):
        final_name = name or tr("Kid","طفل")
        st.session_state["child_name"] = final_name
        st.balloons()
        try:
            greet = tr("Hi, {name}!","مرحباً يا {name}!").format(name=final_name)
            st.session_state["name_greeting_bytes"] = tts_gtts_bytes(greet, lang=get_lang())
            st.session_state["play_name_greeting"] = True
        except Exception:
            st.session_state["name_greeting_bytes"] = b""
            st.session_state["play_name_greeting"] = False
        st.session_state["onboarding_step"] = "age"
        st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

def age_step():
    st.markdown("<div class='kids-ui'>", unsafe_allow_html=True)
    st.subheader(tr("🎂 How old are you?","🎂 كم عمرك؟"))
    st.caption(tr("Tap one","اختر عمرك"))

    row = st.columns(10)
    picked = None
    for i, n in enumerate(range(1,11)):
        with row[i]:
            label = number_chip(n)  # colorful number badge
            if st.button(label, key=f"age_btn_{n}"):
                picked = n
            st.markdown(f"<div style='text-align:center;opacity:.7'>{n}</div>", unsafe_allow_html=True)
    if picked is not None:
        st.session_state["kid_age"] = picked
        st.session_state["onboarding_step"] = "ask"
        st.rerun()

    name = st.session_state.get("child_name", tr("Kid","طفل"))
    st.markdown(f"<div class='wave'>{tr('Hi,','مرحباً،')} {html.escape(name)}!</div>", unsafe_allow_html=True)

    if st.session_state.pop("play_name_greeting", False):
        audio_bytes = st.session_state.pop("name_greeting_bytes", b"")
        if audio_bytes:
            st.audio(audio_bytes, format="audio/mp3")

    st.markdown("</div>", unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# Ask page
# ------------------------------------------------------------------------------
def audio_input_ui():
    if HAS_AUDIO_RECORDER:
        st.caption(tr("🎙️ Record your question","🎙️ سجّل سؤالك"))
        try:
            audio_bytes = audio_recorder(pause_threshold=1.0, sample_rate=16000,
                                         text=tr("Tap to record / stop","اضغط للتسجيل/إيقاف"))
        except Exception:
            audio_bytes = None
        return audio_bytes, "recorder"
    st.caption(tr("📁 Upload a short WAV clip","📁 ارفع ملف WAV قصير"))
    f = st.file_uploader(tr("Choose .wav","اختر .wav"), type=["wav"], label_visibility="collapsed")
    return (f.read() if f else None), "upload"

def explain_three(question: str, answer: str, age: int | None):
    lang_line = _lang_hint()
    def _g(p):
        try:
            return ask_gemini(p).strip()
        except Exception:
            return ask_simple(p)
    picture = _g(
        f"Explain with a visual description for a child age {age or tr('kid','طفل')} to understand: {answer}. "
        f"2–3 cheerful sentences. {lang_line}"
    )
    story = _g(
        f"Tell a short cute story (<=80 words) teaching the idea: {answer}. "
        f"Use simple words. {lang_line}"
    )
    steps = _g(
        f"Explain in 3–5 very short steps to understand: {answer}. "
        f"Return each step on a new line, no numbering. {lang_line}"
    )
    return {"picture": picture, "story": story, "steps": steps}

def render_ask():
    st.markdown("<div class='kids-ui'>", unsafe_allow_html=True)
    st.title(tr("👨‍👧 Ask DAD AI","👨‍👧 اسأل بابا الذكي"))
    step = st.session_state.get("onboarding_step", "name")
    if step == "name":
        name_step(); st.markdown("</div>", unsafe_allow_html=True); return
    if step == "age":
        age_step(); st.markdown("</div>", unsafe_allow_html=True); return

    name = st.session_state.get("child_name", tr("Kid","طفل"))
    age  = st.session_state.get("kid_age")

    q = st.text_input(tr("❓ What do you want to ask?","❓ ما الذي تريد سؤاله؟"), key="ask_q")
    audio_bytes, _src = audio_input_ui()
    if audio_bytes:
        st.audio(audio_bytes, format="audio/wav")
        with st.spinner(tr("Transcribing…","جاري التفريغ الصوتي…")):
            text, err = transcribe_audio(audio_bytes)
        if text:
            st.session_state["ask_q"] = text
            st.success(tr("✅ Added your question above.","✅ أضفنا سؤالك في الأعلى.")); st.rerun()
        else:
            st.error(tr("🛑 Couldn't transcribe your audio.","🛑 تعذّر تفريغ الصوت."))
            if err: st.caption(err)

    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button(tr("✨ Get Answer","✨ احصل على الإجابة")):
            if not q.strip():
                st.info(tr("Please type a question or use the mic/uploader.","من فضلك اكتب سؤالاً أو استخدم الميكروفون/الرفع."))
            else:
                ans = ask_with_book_first(q, age)
                st.session_state["last_q"] = q
                st.session_state["last_a"] = ans
                st.rerun()
    with c2:
        if st.button(tr("🔊 Read Aloud","🔊 قراءة بصوت عالٍ")):
            a = st.session_state.get("last_a","")
            if a:
                try:
                    st.audio(tts_gtts_bytes(a, lang=get_lang()), format="audio/mp3")
                except Exception as e:
                    st.error(f"TTS error: {e}")
            else:
                st.info(tr("Ask something first!","اسأل شيئاً أولاً!"))
    with c3:
        if st.button(tr("🔁 Ask another","🔁 سؤال آخر")):
            for k in ["last_q","last_a"]: st.session_state.pop(k, None)
            st.rerun()

    if st.session_state.get("last_a"):
        st.markdown(tr("#### 🌟 Answer","#### 🌟 الإجابة"))
        st.success(st.session_state["last_a"])
        e3 = explain_three(st.session_state.get("last_q",""), st.session_state["last_a"], age)
        tabs = st.tabs([tr("🖼 Picture","🖼 صورة"), tr("📖 Story","📖 قصة"), tr("🪜 Steps","🪜 خطوات")])
        with tabs[0]:
            st.write(e3["picture"])
        with tabs[1]:
            st.write(e3["story"])
        with tabs[2]:
            render_alpha_steps(e3["steps"])

    st.markdown("</div>", unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# Learning Book (quick PDF QA, not stored)
# ------------------------------------------------------------------------------
def render_learning_book():
    st.title(tr("📚 Learning Book","📚 كتاب التعلم"))
    f = st.file_uploader(tr("Choose a PDF","اختر ملف PDF"), type=["pdf"])
    if f is not None:
        with st.spinner(tr("Extracting text…","جاري استخراج النص…")):
            pdf_bytes = f.read()
            txt, used_ocr = extract_text_any(pdf_bytes)
        if len(txt.strip()) < 40:
            if used_ocr and not OCR_OK:
                st.warning(tr(
                    "I couldn't read enough text. Your PDF looks scanned. Install tesseract-ocr + poppler.",
                    "لم أستطع قراءة نص كافٍ. يبدو أن الملف ممسوح ضوئياً (نحتاج tesseract-ocr + poppler)."
                ))
            else:
                st.warning(tr("Couldn't extract much text from this PDF.","تعذّر استخراج نص كافٍ."))
        else:
            st.success(tr("Book loaded! Ask a question below.","تم تحميل الكتاب! اسأل سؤالك أدناه."))
            st.session_state["learning_book_text"] = txt

    txt = st.session_state.get("learning_book_text","")
    if txt:
        q = st.text_input(tr("Ask about the book:","اسأل عن الكتاب:"))
        if st.button(tr("🔎 Answer from book","🔎 أجب من الكتاب")):
            chunks = chunk_text(txt)
            # lightweight char-ngrams search without model
            try:
                from sklearn.feature_extraction.text import TfidfVectorizer
                vect = TfidfVectorizer(analyzer="char", ngram_range=(3,5))
                X = vect.fit_transform(chunks + [q])
                sims = cosine_similarity(X[-1], X[:-1]).flatten()
                top = [chunks[i] for i in sims.argsort()[::-1][:4]]
                ctx = "\n\n".join(top)
            except Exception:
                ctx = chunks[0][:1000] if chunks else ""
            prompt = f"Use this text to answer the kid shortly.\n\n{ctx}\n\nQ: {q}\n\n{_lang_hint()}"
            ans = ask_simple(prompt)
            st.markdown(tr("#### Answer","#### الإجابة"))
            st.success(ans)

# ------------------------------------------------------------------------------
# Book Store (upload/store/manage + choose active book)
# ------------------------------------------------------------------------------
def render_book_store():
    st.title(tr("📚 Book Store","📚 مخزن الكتب"))
    st.caption(tr("Upload Arabic/English PDFs. We'll index & store them so kids' questions use the active book first.",
                  "ارفع ملفات PDF (عربي/إنجليزي). سنفهرسها ونستخدم الكتاب النشط للإجابة أولاً."))

    # Upload
    f = st.file_uploader(tr("Upload a PDF to store","ارفع ملف PDF للتخزين"), type=["pdf"])
    if f is not None:
        with st.spinner(tr("Reading & indexing…","جاري القراءة والفهرسة…")):
            pdf_bytes = f.read()
            txt, used_ocr = extract_text_any(pdf_bytes)
            if len(txt.strip()) >= 40:
                info = build_book_index(txt, f.name)
                st.success(tr("Book stored and indexed!","تم تخزين الكتاب وفهرسته!"))
            else:
                if used_ocr and not OCR_OK:
                    st.error(tr("Looks scanned; OCR not available on server.","الملف ممسوح ضوئياً، و OCR غير متاح على الخادم."))
                else:
                    st.error(tr("Couldn't extract useful text.","تعذّر استخراج نص مفيد."))

    # List + choose active + delete
    books = list_books()
    if not books:
        st.info(tr("No books stored yet.","لا توجد كتب بعد."))
        return

    active = st.session_state.get("active_book_id")
    for meta in books:
        with st.expander(meta["title"], expanded=False):
            book_id = meta["book_id"]
            npy = meta.get("embeddings_path")
            st.write(tr("Chunks","القطع النصية"), len(meta.get("chunks", [])))
            st.write(tr("Embeddings path","مسار المتجهات:"), npy)
            col1, col2 = st.columns([1,1])
            with col1:
                if st.button(tr("Use","استخدم"), key=f"use_{book_id}"):
                    st.session_state["active_book_id"] = book_id
                    st.success(tr(f"Active book: {meta['title']}", f"الكتاب النشط: {meta['title']}"))
            with col2:
                if st.button(tr("Delete","حذف"), key=f"del_{book_id}"):
                    if active == book_id:
                        st.session_state.pop("active_book_id", None)
                    delete_book(book_id)
                    st.rerun()

    if st.session_state.get("active_book_id"):
        meta = load_book(st.session_state["active_book_id"])
        if meta:
            st.success(tr(f"Active book: {meta['title']}", f"الكتاب النشط: {meta['title']}"))

# ------------------------------------------------------------------------------
# Placeholders for other tabs (you can wire your own modules here)
# ------------------------------------------------------------------------------
def render_animal_fun():
    st.title(tr("🐾 Animal Fun","🐾 مرح مع الحيوانات"))
    animal = st.text_input(tr("Animal name (e.g., cat, dog, lion)","اسم الحيوان (مثلاً: قط، كلب، أسد)"))
    if st.button(tr("🔊 Play Animal Sound","🔊 شغّل صوت الحيوان")):
        st.info(tr(f"Pretend we're playing the sound of {animal} 🐾","سنُشغّل صوت الحيوان افتراضياً 🐾"))

def render_quiz():
    st.title(tr("🧠 Quiz Fun","🧠 مسابقة ممتعة"))
    st.info(tr("Wire your quiz here.","اربط مسابقتك هنا."))

def render_scoreboard():
    st.title(tr("📊 Scoreboard","📊 لوحة النتائج"))
    st.info(tr("Show your saved scores here.","اعرض نتائجك هنا."))

def render_draw_guess():
    st.title(tr("🎨 Draw & Guess","🎨 ارسم وخمّن"))
    st.info(tr("Plug your drawable canvas game here.","أضف لعبة الرسم هنا."))

# ------------------------------------------------------------------------------
# Router
# ------------------------------------------------------------------------------
if tab == tr("💬 Ask DAD AI","💬 اسأل بابا الذكي"):
    render_ask()
elif tab == tr("📚 Learning Book","📚 كتاب التعلم"):
    render_learning_book()
elif tab == tr("📚 Book Store (choose & manage)","📚 مخزن الكتب (اختيار وإدارة)"):
    render_book_store()
elif tab == tr("🐾 Animal Fun","🐾 مرح مع الحيوانات"):
    render_animal_fun()
elif tab == tr("🧠 Quiz Fun","🧠 مسابقة ممتعة"):
    render_quiz()
elif tab == tr("📊 Scoreboard","📊 النتائج"):
    render_scoreboard()
elif tab == tr("🎨 Draw & Guess","🎨 ارسم وخمّن"):
    render_draw_guess()
