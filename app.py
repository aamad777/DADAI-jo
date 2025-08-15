# app.py — Ask DAD AI (Classic + Neon UI + Arabic + Robust PDF + Book-first QA)

import os, json, threading, queue, random, html, tempfile, re
from datetime import datetime
from io import BytesIO
from typing import List, Tuple

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
from kid_feedback import send_email_to_dad
from quiz_game import get_quiz_question
from quiz_sounds import play_correct_sound, play_wrong_sound, play_win_sound
from quiz_scoreboard import log_score, show_scoreboard
from streamlit_drawable_canvas import st_canvas
from gemini_ai import classify_sketch, fetch_animal_photo, ask_gemini

# ---------------------------------------------------------------
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY", "")
client = OpenAI(api_key=openai_api_key) if openai_api_key else None

# ===== Language & helpers =====================================================
LANGS = {"English": "en", "العربية": "ar"}

def init_lang_state():
    if "lang" not in st.session_state:
        st.session_state["lang"] = "en"
init_lang_state()

def get_lang() -> str:
    return st.session_state.get("lang", "en")

def is_ar() -> bool:
    return get_lang().startswith("ar")

# translator helper (renamed from "_" to avoid collisions)
def tr(en: str, ar: str) -> str:
    return ar if is_ar() else en

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
        lang_code = "ar" if is_ar() else "en-US"
        text = r.recognize_google(audio, language=lang_code)
        out_q.put((text, None))
    except Exception as e:
        out_q.put((None, f"STT failed: {e}"))

def transcribe_audio(audio_bytes: bytes):
    # Whisper first
    if os.getenv("OPENAI_API_KEY"):
        try:
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

/* Compliments */
.compliment-row { display:flex; flex-wrap:wrap; gap:8px; margin: 8px 0 2px 0; }
.comp-chip {
  display:inline-flex; align-items:center; gap:8px; padding:10px 12px; border-radius:14px;
  background:linear-gradient(135deg,#e9d5ff,#bfdbfe); color:#0b1324; font-weight:900;
  box-shadow:0 8px 16px rgba(0,0,0,.08); border:1.5px solid rgba(255,255,255,.6);
}
.float-emoji { position:relative; display:inline-block; }
.float-emoji:before { content:"🎉"; position:absolute; left:-6px; top:-10px; opacity:.85; animation: floatUp 1.6s ease-in-out infinite; }
@keyframes floatUp { 0%{ transform: translateY(4px); opacity:.8} 50%{ transform: translateY(-4px); opacity:1} 100%{ transform: translateY(4px); opacity:.8} }

/* Colorful alphabet bullets */
.alpha-chip {
  display:inline-flex; align-items:center; justify-content:center;
  width:28px; height:28px; border-radius:50%;
  margin-right:8px; font-weight:900; color:#0b1324;
  background: linear-gradient(135deg,var(--g1),var(--g2));
  box-shadow:0 4px 10px rgba(0,0,0,.12);
}
.alpha-row { display:flex; align-items:flex-start; gap:8px; margin:6px 0; }

/* For age chips */
.age-chip {
  display:inline-flex; align-items:center; justify-content:center;
  width:28px; height:28px; border-radius:50%;
  font-weight:900; color:#0b1324;
  background: linear-gradient(135deg,var(--g1),var(--g2));
  box-shadow:0 4px 10px rgba(0,0,0,.12);
  margin-bottom:6px;
}
</style>
""", unsafe_allow_html=True)

# RTL tweaks for Arabic
if is_ar():
    st.markdown("""
    <style>
      html, body, [data-testid="stAppViewContainer"] * { direction: rtl; text-align: right; }
      .neon-input input, .msg, .stTextInput, .stButton { text-align: right !important; }
      .alpha-row { flex-direction: row-reverse; }
      .alpha-chip { margin-right:0; margin-left:8px; }
    </style>
    """, unsafe_allow_html=True)

# ===== Sidebar ================================================================
st.sidebar.title("📚 DAD AI Navigation")
lang_choice = st.sidebar.selectbox("Language / اللغة", list(LANGS.keys()),
                                   index=0 if get_lang()=="en" else 1)
st.session_state["lang"] = LANGS[lang_choice]

tab = st.sidebar.radio(tr("Choose a tab:", "اختر صفحة:"),
    [
        tr("💬 Ask DAD AI", "💬 اسأل بابا الذكي"),
        tr("🐾 Animal Fun", "🐾 مرح مع الحيوانات"),
        tr("🛠️ Dad's Dashboard", "🛠️ لوحة تحكم الأب"),
        tr("📚 Learning Book", "📚 كتاب التعلم"),
        tr("🧠 Quiz Fun", "🧠 مسابقة ممتعة"),
        tr("📊 Scoreboard", "📊 لوحة النتائج"),
        tr("🎨 Draw & Guess (Gemini)", "🎨 ارسم وخمّن (Gemini)")
    ]
)

# Toggle UI style
ui_style = st.sidebar.selectbox(tr("🎨 UI style", "🎨 نمط الواجهة"), ["Classic", "Neon"], index=0)

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

# ===== Colorful alphabet helpers =============================================
ALPHA_COLORS = [
    ("#fde68a","#fca5a5"), ("#bbf7d0","#86efac"), ("#c7d2fe","#93c5fd"),
    ("#fbcfe8","#fda4af"), ("#bae6fd","#93c5fd"), ("#d1fae5","#a7f3d0"),
]

def render_alpha_steps(text_block: str):
    """Render steps as A) B) C) with colorful round chips."""
    if not text_block.strip():
        return
    lines = [ln.strip() for ln in text_block.splitlines() if ln.strip()]
    cleaned = []
    for ln in lines:
        ln = ln.lstrip("-•*").strip()
        while len(ln) > 1 and (ln[0].isdigit() or ln[0].isalpha()) and ln[1] in [")", ".", "］", "】", "）", "．"]:
            ln = ln[2:].strip()
        cleaned.append(ln)
    for i, ln in enumerate(cleaned):
        g1, g2 = random.choice(ALPHA_COLORS)
        chip = f"<span class='alpha-chip' style='--g1:{g1};--g2:{g2}'>{chr(65+i)}</span>"
        st.markdown(f"<div class='alpha-row'>{chip}<div>{html.escape(ln)}</div></div>", unsafe_allow_html=True)

# ===== Model wrapper ==========================================================
def _lang_hint():
    return "Respond in Arabic (Modern Standard Arabic) with very simple words." if is_ar() \
           else "Respond in English with very simple words."

def ask_with_context(question: str, category: str | None, age: int | None) -> str:
    answers = load_answers()
    for k, v in answers.items():
        if k.lower() in question.lower():
            return v
    topic = category or "General"
    age_text = f"{age}" if age else tr("kid","طفل")
    instruction = (
        f"Please answer like a kind teacher for a child age {age_text}. "
        f"Topic: {topic}. Keep it short, clear, and fun. Use simple words. "
        f"{_lang_hint()}"
    )
    try:
        return ask_gemini(f"{instruction}\nQuestion: {question}")
    except Exception:
        pass
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
            return tr(f"Sorry, I couldn't answer right now: {e}", f"عذراً، لا أستطيع الإجابة الآن: {e}")
    return tr("Sorry, I couldn't answer right now.", "عذراً، لا أستطيع الإجابة الآن.")

# ===== Audio input (mic if available, else WAV upload) ========================
def audio_input_ui():
    if HAS_AUDIO_RECORDER:
        st.caption(tr("🎙️ Record your question", "🎙️ سجّل سؤالك"))
        try:
            audio_bytes = audio_recorder(pause_threshold=1.0, sample_rate=16000,
                                         text=tr("Tap to record / stop","اضغط للتسجيل/إيقاف"))
        except Exception:
            audio_bytes = None
        return audio_bytes, "recorder"
    else:
        st.caption(tr("📁 Upload a short WAV clip (mic not available on this server)",
                     "📁 ارفع ملف WAV قصير (الميكروفون غير متاح على هذا الخادم)"))
        file = st.file_uploader(tr("Choose a .wav file","اختر ملف .wav"),
                                type=["wav"], accept_multiple_files=False, label_visibility="collapsed")
        if file:
            return file.read(), "upload"
        return None, "upload"

# ===== Name bubbles (animated) ================================================
BUBBLE_COLORS = [
    ("#a7f3d0", "#86efac"), ("#93c5fd", "#bfdbfe"), ("#fbcfe8", "#fda4af"),
    ("#fde68a", "#fca5a5"), ("#c7d2fe", "#93c5fd"), ("#fdba74", "#fef3c7"),
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

# ===== Arabic-friendly extraction + embeddings (self-contained) ===============
# (Same logic you would normally put in a separate module)
ARABIC_DIACRITICS = re.compile(r"[\u0610-\u061A\u064B-\u065F\u06D6-\u06ED]")

def normalize_arabic(s: str) -> str:
    s = ARABIC_DIACRITICS.sub("", s)
    s = s.replace("ـ", "")
    s = re.sub("[إأآا]", "ا", s)
    s = s.replace("ى", "ي").replace("ئ", "ي").replace("ؤ", "و").replace("ة", "ه")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def extract_text_pymupdf_blocks(pdf_bytes: bytes) -> str:
    try:
        import fitz
    except Exception:
        return ""
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        pages = []
        for p in doc:
            blocks = p.get_text("blocks")
            blocks.sort(key=lambda b: (round(b[1], 1), round(b[0], 1)))  # y then x
            page = []
            for b in blocks:
                t = (b[4] or "").strip()
                if t:
                    page.append(t)
            pages.append("\n".join(page))
        return "\n\n".join(pages)
    except Exception:
        return ""

def extract_text_pdfminer(pdf_bytes: bytes) -> str:
    try:
        from pdfminer.high_level import extract_text
    except Exception:
        return ""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(pdf_bytes); tmp.flush()
            return extract_text(tmp.name) or ""
    except Exception:
        return ""

def ocr_with_ocrmypdf_then_extract(pdf_bytes: bytes) -> str:
    try:
        import subprocess, fitz
    except Exception:
        return ""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as src:
            src.write(pdf_bytes); src.flush()
            dst = src.name.replace(".pdf", ".ocr.pdf")
        cmd = ["ocrmypdf", "--force-ocr", "--deskew", "--clean", "--ocr-image-dpi", "300", "--optimize", "1", src.name, dst]
        subprocess.run(cmd, check=True, capture_output=True)
        doc = fitz.open(dst)
        return "\n".join((p.get_text("text") or "") for p in doc)
    except Exception:
        return ""

def ocr_pdf_images(pdf_bytes: bytes, lang: str="ar") -> str:
    try:
        import pytesseract
        from pdf2image import convert_from_bytes
        from PIL import Image, ImageOps, ImageFilter
    except Exception:
        return ""
    tess_lang = "ara+eng" if lang.startswith("ar") else "eng"
    poppler_path = os.getenv("POPPLER_PATH", None)
    pages = convert_from_bytes(pdf_bytes, dpi=400, poppler_path=poppler_path) \
            if poppler_path else convert_from_bytes(pdf_bytes, dpi=400)
    out = []
    for img in pages:
        g = ImageOps.grayscale(img)
        g = ImageOps.autocontrast(g)
        g = g.filter(ImageFilter.UnsharpMask(radius=1.2, percent=150, threshold=3))
        cfg = "--oem 3 --psm 6 -c preserve_interword_spaces=1"
        out.append(pytesseract.image_to_string(g, lang=tess_lang, config=cfg))
    return "\n".join(out)

def robust_extract_text(pdf_bytes: bytes, lang: str="ar") -> Tuple[str, str]:
    t1 = extract_text_pymupdf_blocks(pdf_bytes)
    if len(normalize_arabic(t1)) > 120:
        return t1, "pymupdf-blocks"
    t2 = extract_text_pdfminer(pdf_bytes)
    if len(normalize_arabic(t2)) > 120:
        return t2, "pdfminer"
    t3 = ocr_with_ocrmypdf_then_extract(pdf_bytes)
    if len(normalize_arabic(t3)) > 120:
        return t3, "ocrmypdf"
    t4 = ocr_pdf_images(pdf_bytes, lang=lang)
    return t4, "pytesseract"

def chunk_text(text: str, chunk_chars: int = 1200, overlap: int = 250) -> List[str]:
    text = re.sub(r"\s+", " ", text).strip()
    chunks, i = [], 0
    while i < len(text):
        end = min(len(text), i + chunk_chars)
        cut = text[i:end]
        dot = max(cut.rfind("۔"), cut.rfind("."))  # try to stop on sentence end
        if dot >= 0 and end - (i + dot) < 300:
            end = i + dot + 1
        chunks.append(text[i:end])
        i = max(end - overlap, i + 1)
    return chunks

class LocalBookIndex:
    def __init__(self):
        self.model = None
        self.index = None
        self.chunks = []

    def _load_model(self):
        if self.model is None:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

    def build(self, raw_text: str):
        self._load_model()
        self.chunks = chunk_text(raw_text)
        texts = [normalize_arabic(c) for c in self.chunks]
        embs = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        import faiss
        d = embs.shape[1]
        self.index = faiss.IndexFlatIP(d)
        self.index.add(embs.astype("float32"))

    def search_text(self, query: str, top_k: int = 6) -> str:
        if self.index is None:
            return ""
        self._load_model()
        import faiss
        qv = self.model.encode([normalize_arabic(query)], normalize_embeddings=True).astype("float32")
        D, I = self.index.search(qv, top_k)
        parts = []
        for idx in I[0]:
            if idx >= 0:
                parts.append(self.chunks[int(idx)])
        return "\n\n".join(parts)

# ===== Book-first QA ==========================================================
def answer_kid_question_with_book_first(question: str, category: str | None, age: int | None) -> str:
    passages = ""
    book_ix = st.session_state.get("book_index")
    if book_ix is not None:
        try:
            passages = book_ix.search_text(question, top_k=6)
        except Exception:
            passages = ""

    if passages.strip():
        prompt = (
            f"Use the following book excerpts to answer a child's question.\n\n"
            f"EXCERPTS:\n{passages}\n\n"
            f"QUESTION: {question}\n\n"
            f"RULES: {_lang_hint()} Keep it short, clear, and fun. "
            f"If the answer isn’t in the excerpts, say you aren’t sure."
        )
        try:
            ans = ask_gemini(prompt).strip()
            if ans:
                return ans
        except Exception:
            pass
        if client:
            try:
                resp = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role":"user","content":prompt}],
                    temperature=0.4, max_tokens=220
                )
                ans = (resp.choices[0].message.content or "").strip()
                if ans:
                    return ans
            except Exception:
                pass

    return ask_with_context(question, category, age)

# ===== Onboarding: name -> age -> ask ========================================
def name_step():
    st.markdown("<div class='kids-ui'>", unsafe_allow_html=True)
    st.subheader(tr("🧩 What's your name?", "🧩 ما اسمك؟"))
    current_name = st.session_state.get("kid_name", "")
    typed = st.text_input(tr("Type your name here:","اكتب اسمك هنا:"), value=current_name)
    if typed != current_name:
        st.session_state["kid_name"] = typed
    name = (st.session_state.get("kid_name","") or "").strip()
    st.caption(tr("Preview","معاينة"))
    st.markdown(bubble_name_html(name), unsafe_allow_html=True)
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

def _age_chip(n: int):
    g1, g2 = random.choice(ALPHA_COLORS)
    st.markdown(f"<div class='age-chip' style='--g1:{g1};--g2:{g2}'>{n}</div>", unsafe_allow_html=True)

def age_step():
    st.markdown("<div class='kids-ui'>", unsafe_allow_html=True)
    st.subheader(tr("🎂 How old are you?","🎂 كم عمرك؟"))
    st.caption(tr("Tap one","اختر عمرك"))
    row = st.columns(10)
    picked = None
    for i, n in enumerate(range(1,11)):
        with row[i]:
            _age_chip(n)
            if st.button(str(n), key=f"age_{n}"):
                picked = n
    if picked is not None:
        st.session_state["kid_age"] = picked
        st.session_state["age_compliments_list"] = AGE_COMPLIMENTS_3.get(picked, [tr("🎈 Awesome age!","🎈 عمر رائع!")])
        st.session_state["age_comp_index"] = 0
        st.session_state["age_celebrate_msg"] = "age_ready"
        st.session_state["onboarding_step"] = "ask"
        st.rerun()
    name = st.session_state.get("child_name", tr("Kid","طفل"))
    st.markdown(f"<div class='wave'>{tr('Hi,','مرحباً،')} {html.escape(name)}!</div>", unsafe_allow_html=True)
    st.markdown(bubble_name_html(name), unsafe_allow_html=True)
    if st.session_state.pop("play_name_greeting", False):
        audio_bytes = st.session_state.pop("name_greeting_bytes", b"")
        if audio_bytes:
            st.audio(audio_bytes, format="audio/mp3")
    st.markdown("</div>", unsafe_allow_html=True)

def render_category_picker():
    st.markdown(tr("#### 🎒 Pick a topic","#### 🎒 اختر موضوعاً"))
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
                if st.button((tr("Choose","اختر") + f" {k}"), key=f"choose_{k}"):
                    st.session_state["topic_category"] = k
                    st.rerun()

def render_idea_chips(category: str):
    ideas = CATEGORIES.get(category, {}).get("ideas", [])
    if not ideas: return
    st.markdown(tr("##### Try one of these:","##### جرّب أحد هذه الأسئلة:"))
    chip_cols = st.columns(min(6, len(ideas)))
    for i, idea in enumerate(ideas):
        with chip_cols[i % len(chip_cols)]:
            if st.button(idea, key=f"idea_{category}_{i}"):
                st.session_state["child_question"] = idea
                st.rerun()

def _explain_three_ways(base_q: str, base_a: str, age: int | None, category: str | None):
    if "explain3" in st.session_state and st.session_state.get("explain3_q") == base_q:
        return st.session_state["explain3"]
    age_text = f"{age}" if age else tr("kid","طفل")
    topic = category or "General"
    def _gen(prompt: str) -> str:
        try:
            return ask_gemini(prompt).strip()
        except Exception:
            pass
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
    lang_line = _lang_hint()
    picture = _gen(
        (f"Explain with a visual description for a child age {age_text} about this topic ({topic}). "
         f"Describe a simple picture (text-only) to understand this answer: {base_a}. "
         f"Keep it to 2-3 cheerful sentences. {lang_line}")
    )
    story = _gen(
        (f"Tell a short, cute story (<=80 words) for a child age {age_text} that teaches the idea in this answer: {base_a}. "
         f"Use simple words and a friendly tone. {lang_line}")
    )
    steps = _gen(
        (f"Explain in 3–5 very short steps for a child age {age_text} to understand the idea behind this answer: {base_a}. "
         f"Return each step on a new line, no numbering. {lang_line}")
    )
    st.session_state["explain3"] = {"picture": picture, "story": story, "steps": steps}
    st.session_state["explain3_q"] = base_q
    return st.session_state["explain3"]

def ask_step():
    st.markdown("<div class='kids-ui'>", unsafe_allow_html=True)
    name = st.session_state.get("child_name", tr("Kid","طفل"))
    age = st.session_state.get("kid_age")
    category = st.session_state.get("topic_category")
    msg = st.session_state.pop("age_celebrate_msg", None)
    if msg == "age_ready":
        try: play_win_sound()
        except Exception: pass
        st.balloons()
        comps = st.session_state.get("age_compliments_list", [])
        idx = st.session_state.get("age_comp_index", 0)
        if comps:
            rot = comps[idx:] + comps[:idx]
            show = rot[:3] if len(rot) >= 3 else rot
            st.markdown(tr("#### 🎉 You're awesome!","#### 🎉 أنت رائع!"))
            st.markdown("<div class='compliment-row'>", unsafe_allow_html=True)
            for c in show:
                st.markdown(f"<div class='comp-chip'><span class='float-emoji'></span>{html.escape(c)}</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
            if st.button(tr("🔄 Another compliment","🔄 مجاملة أخرى")):
                st.session_state["age_comp_index"] = (idx + 1) % max(1, len(comps))
                st.session_state["age_celebrate_msg"] = "age_ready"
                st.rerun()
    st.markdown(tr("### 👋 Hello,","### 👋 أهلاً،") + f" **{name}**" + (f" — age {age}" if age else ""))
    if not category:
        render_category_picker()
    else:
        cfg = CATEGORIES[category]
        st.markdown(
            f"<div style='background:linear-gradient(135deg,{cfg['colors'][0]},{cfg['colors'][1]});"
            f"border-radius:14px;padding:10px 14px;color:#0b1324;font-weight:900;margin:6px 0;'>"
            f"{cfg['emoji']} {tr('Topic:','الموضوع:')} {category}</div>", unsafe_allow_html=True
        )
        c1, c2, c3 = st.columns([1,1,1])
        with c1:
            if st.button(tr("🔄 Change topic","🔄 تغيير الموضوع")):
                st.session_state.pop("topic_category", None); st.rerun()
        with c2:
            if st.button(tr("🎯 More ideas","🎯 اقتراحات أخرى")):
                random.shuffle(CATEGORIES[category]["ideas"]); st.rerun()
        with c3:
            if st.button(tr("✨ Surprise me","✨ فاجئني")):
                rand_cat = random.choice(list(CATEGORIES.keys()))
                st.session_state["topic_category"] = rand_cat
                idea = random.choice(CATEGORIES[rand_cat]["ideas"])
                st.session_state["prefill_child_question"] = idea
                st.rerun()
        render_idea_chips(category)
    default_q = st.session_state.pop("prefill_child_question", None)
    if default_q is not None:
        question = st.text_input(tr("❓ What do you want to ask?","❓ ما الذي تريد سؤاله؟"),
                                 value=default_q, key="ask_input")
    else:
        question = st.text_input(tr("❓ What do you want to ask?","❓ ما الذي تريد سؤاله؟"), key="ask_input")
    audio_bytes, src = audio_input_ui()
    if audio_bytes:
        st.audio(audio_bytes, format="audio/wav")
        with st.spinner(tr("Transcribing…","جاري التفريغ الصوتي…")):
            text, err = transcribe_audio(audio_bytes)
        if text:
            st.session_state['prefill_child_question'] = text
            st.success(tr("✅ Added your question above.","✅ أضفنا سؤالك في الأعلى.")); st.rerun()
        else:
            st.error(tr("🛑 Couldn't transcribe your audio.","🛑 تعذّر تفريغ الصوت."))
            if err: st.caption(err)
    c1, c2, c3 = st.columns([1,1,1])
    with c1:
        if st.button(tr("✨ Get Answer","✨ احصل على الإجابة"), use_container_width=True):
            if not question.strip():
                st.info(tr("Please type a question or use the mic/uploader.","من فضلك اكتب سؤالاً أو استخدم الميكروفون/الرفع."))
            else:
                answer = answer_kid_question_with_book_first(
                    question.strip(),
                    st.session_state.get("topic_category"),
                    age
                )
                st.session_state["last_answer"] = answer
                st.session_state["last_question"] = question.strip()
                st.session_state["just_answered"] = True
                save_qa_log(name, question.strip(), answer)
                st.rerun()
    with c2:
        if st.button(tr("🔊 Read Aloud","🔊 قراءة بصوت عالٍ"), use_container_width=True):
            answer = st.session_state.get("last_answer","")
            if answer:
                try:
                    st.audio(tts_gtts_bytes(answer, lang=get_lang()), format="audio/mp3")
                except Exception as e:
                    st.error(f"TTS error: {e}")
            else:
                st.info(tr("Ask something first!","اسأل شيئاً أولاً!"))
    with c3:
        if st.button(tr("🔁 Ask another","🔁 سؤال آخر"), use_container_width=True):
            st.session_state.pop("last_answer", None); st.session_state.pop("last_question", None); st.rerun()
    if st.session_state.get("last_answer"):
        if st.session_state.pop("just_answered", False):
            st.balloons()
            try: play_win_sound()
            except Exception: pass
        st.markdown(tr("#### 🌟 Answer","#### 🌟 الإجابة"))
        st.success(st.session_state["last_answer"])
        st.markdown(tr("#### Explain 3 Ways","#### اشرح بثلاث طرق"))
        tabs = st.tabs([tr("🖼 Picture","🖼 صورة"), tr("📖 Story","📖 قصة"), tr("🪜 Steps","🪜 خطوات")])
        e3 = _explain_three_ways(st.session_state.get("last_question",""),
                                 st.session_state["last_answer"], age,
                                 st.session_state.get("topic_category"))
        with tabs[0]:
            st.write(e3.get("picture",""))
            if st.button(tr("🔁 Regenerate picture explain","🔁 إعادة توليد وصف الصورة")):
                st.session_state.pop("explain3", None); st.rerun()
        with tabs[1]:
            st.write(e3.get("story",""))
            if st.button(tr("🔁 Regenerate story","🔁 إعادة توليد القصة")):
                st.session_state.pop("explain3", None); st.rerun()
        with tabs[2]:
            render_alpha_steps(e3.get("steps",""))
            if st.button(tr("🔁 Regenerate steps","🔁 إعادة توليد الخطوات")):
                st.session_state.pop("explain3", None); st.rerun()
        st.markdown(tr("#### Did you understand it?","#### هل فهمت الإجابة؟"))
        y, n = st.columns(2)
        with y:
            if st.button(tr("👍 I understand","👍 فهمت"), key="understand_yes", use_container_width=True):
                st.info(tr("Awesome! Want to try the Quiz tab too?","رائع! هل تريد تجربة تبويب المسابقة أيضاً؟"))
        with n:
            if st.button(tr("✉️ Email Dad (I don't understand)","✉️ أرسل بريداً للأب (لم أفهم)"),
                         key="understand_no", use_container_width=True):
                with st.spinner(tr("Sending email to Dad...","جاري إرسال البريد إلى الأب...")):
                    ok, msg = send_email_to_dad(child_name=name,
                                                question=st.session_state.get("last_question",""),
                                                answer=st.session_state["last_answer"])
                if ok:
                    st.success(tr("Email sent to Dad ✅","تم إرسال البريد إلى الأب ✅")); st.caption(msg)
                else:
                    st.error(tr("Couldn't send email ❌","تعذّر إرسال البريد ❌")); st.code(msg, language="text")
    st.markdown("</div>", unsafe_allow_html=True)

# ===== NEON THEME =============================================================
NEON_CSS = """
<style>
:root{ --neon:#3DF0A5; --neon2:#5BB9FF; --bg:#020617; --glass: rgba(15,23,42,.55);
  --txt:#E6F1FF; --dim:#90A4B4; }
.neon-bg{ position:relative; background:radial-gradient(ellipse at top,#0b1b33 0%,#020617 60%);
  padding:8px; border-radius:24px; box-shadow:0 0 40px rgba(61,240,165,.08) inset; }
.neon-grid:before{ content:""; position:absolute; inset:-20px;
  background: radial-gradient(2px 2px at 20px 20px, rgba(93,185,255,.7), transparent 60%) 0 0/36px 36px,
              radial-gradient(2px 2px at 10px 10px, rgba(61,240,165,.6), transparent 60%) 0 0/30px 30px;
  filter: blur(.4px); opacity:.35; animation: drift 16s linear infinite; }
@keyframes drift{ 0%{transform:translateY(0)} 100%{transform:translateY(-36px)} }
.neon-header{ display:flex; align-items:center; gap:10px; padding:10px 16px; border-radius:16px;
  background:linear-gradient(90deg, rgba(61,240,165,.18), rgba(91,185,255,.18));
  border:2px solid rgba(91,185,255,.35);
  box-shadow:0 0 16px rgba(61,240,165,.25), inset 0 0 20px rgba(91,185,255,.12);
  color:var(--txt); font-weight:900; letter-spacing:.5px; }
.neon-chip{font-size:14px;color:var(--dim);margin-left:auto}
.neon-window{ margin-top:10px; border-radius:26px; padding:16px; min-height:360px;
  background:linear-gradient(180deg, rgba(10,19,39,.85), rgba(6,12,25,.85));
  border:2px solid rgba(61,240,165,.35); box-shadow:0 0 18px rgba(61,240,165,.25), inset 0 0 28px rgba(91,185,255,.14);
  color:var(--txt); }
.msg{max-width:80%; padding:10px 14px; border-radius:14px; margin:8px 0; display:inline-block}
.msg.user{background:rgba(91,185,255,.18); border:1px solid rgba(91,185,255,.45)}
.msg.bot{background:rgba(61,240,165,.16); border:1px solid rgba(61,240,165,.45)}
.row{display:flex;gap:8px} .right{justify-content:flex-end}
.neon-input{ margin-top:12px; display:flex; gap:8px; align-items:center; background:var(--glass);
  border:2px solid rgba(91,185,255,.35); border-radius:18px; padding:8px; box-shadow: inset 0 0 14px rgba(91,185,255,.10); }
.neon-input input{ background:transparent!important; color:var(--txt)!important; }
.neon-send button{ width:64px; height:44px; border-radius:12px; border:2px solid rgba(61,240,165,.55)!important;
  background:linear-gradient(180deg, rgba(61,240,165,.25), rgba(91,185,255,.20))!important; color:#0d1b2a!important;
  font-weight:900; box-shadow:0 0 18px rgba(61,240,165,.28); }
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
            ans = answer_kid_question_with_book_first(q.strip(), st.session_state.get("topic_category"), age)
        except Exception:
            ans = tr("Sorry, couldn't answer right now.","عذراً، لا أستطيع الإجابة الآن.")
        st.session_state.neon_chat.append(("bot", ans))
        st.session_state["last_answer"] = ans
        st.session_state["last_question"] = q.strip()
        st.session_state["child_question"] = ""
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

# ===== PDF Learning Book (Arabic/English) =====================================
def render_learning_book_tab_local():
    st.write(tr("Upload a PDF book (English or Arabic). We'll search it first before using AI.",
               "ارفع كتاب PDF (عربي أو إنجليزي). سنبحث فيه أولاً قبل استخدام الذكاء الاصطناعي."))
    file = st.file_uploader(tr("Choose a PDF", "اختر ملف PDF"), type=["pdf"])
    if file is not None:
        with st.spinner(tr("Extracting text…","جاري استخراج النص…")):
            pdf_bytes = file.read()
            txt, method = robust_extract_text(pdf_bytes, get_lang())
        if len(normalize_arabic(txt)) < 200:
            st.warning(tr(
                "I couldn't read enough text from this PDF. If it is scanned, we enabled OCR on the server, but some files still need higher quality.",
                "تعذّر استخراج نص كافٍ من هذا الملف. إذا كان ممسوحاً ضوئياً فقد تحتاج لنسخة أوضح."
            ))
        else:
            st.success(tr(f"Book loaded using: {method}", f"تم تحميل الكتاب بطريقة: {method}"))
            st.session_state["learning_book_text"] = txt
            # Build/refresh an embedding index for this book
            st.session_state["book_index"] = LocalBookIndex()
            st.session_state["book_index"].build(txt)
            st.session_state["active_book_name"] = file.name

    book_text = st.session_state.get("learning_book_text","")
    if book_text:
        q = st.text_input(tr("Ask about the book:","اسأل عن الكتاب:"))
        if st.button(tr("🔎 Answer from book","🔎 أجب من الكتاب")):
            if not q.strip():
                ans = tr("Please type a question.","من فضلك اكتب سؤالاً.")
            else:
                passages = st.session_state["book_index"].search_text(q, top_k=6)
                if passages.strip():
                    prompt = (f"Use these passages to answer the child's question.\n\n"
                              f"{passages}\n\n"
                              f"Q: {q}\n\n{_lang_hint()} Keep it short and friendly.")
                    try:
                        ans = ask_gemini(prompt)
                    except Exception:
                        if client:
                            try:
                                resp = client.chat.completions.create(
                                    model="gpt-4o-mini",
                                    messages=[{"role":"user","content":prompt}],
                                    temperature=0.4, max_tokens=220
                                )
                                ans = (resp.choices[0].message.content or "").strip()
                            except Exception as e:
                                ans = tr(f"Book QA error: {e}", f"خطأ في سؤال الكتاب: {e}")
                        else:
                            ans = tr("Sorry, I couldn't answer right now.", "عذراً، لا أستطيع الإجابة الآن.")
                else:
                    ans = tr("I'm not sure. The text seems unrelated or too short.",
                             "لست متأكداً. يبدو أن النص غير متعلق أو قصير جداً.")
            st.markdown(tr("#### Answer","#### الإجابة"))
            st.success(ans)
            if st.button(tr("🔊 Read Aloud","🔊 قراءة بصوت عالٍ")):
                try:
                    st.audio(tts_gtts_bytes(ans, lang=get_lang()), format="audio/mp3")
                except Exception as e:
                    st.error(f"TTS error: {e}")

# ========================= MAIN TABS ==========================================
if tab == tr("💬 Ask DAD AI", "💬 اسأل بابا الذكي"):
    st.title(tr("👨‍👧 Ask DAD AI","👨‍👧 اسأل بابا الذكي"))
    step = st.session_state.get("onboarding_step", "name")
    if step == "name":
        name_step()
    elif step == "age":
        age_step()
    else:
        if "child_name" not in st.session_state:
            st.session_state["child_name"] = (st.session_state.get("kid_name","") or tr("Kid","طفل")).strip()
        if ui_style == "Neon":
            render_neon_chat_ui()
        else:
            ask_step()

elif tab == tr("🐾 Animal Fun", "🐾 مرح مع الحيوانات"):
    st.title(tr("🐾 Animal Fun", "🐾 مرح مع الحيوانات"))
    st.write(tr("Type an animal name to play a sound, or generate a fun drawing!",
               "اكتب اسم حيوان لتشغيل صوته أو توليد رسمة لطيفة!"))
    col1, col2 = st.columns(2)
    with col1:
        animal = st.text_input(tr("Animal name (e.g., cat, dog, lion)","اسم الحيوان (مثلاً: قط، كلب، أسد)"))
        if st.button(tr("🔊 Play Animal Sound","🔊 شغّل صوت الحيوان")):
            if animal.strip(): play_animal_sound(animal.strip().lower())
            else: st.info(tr("Please enter an animal name.","أدخل اسم الحيوان."))
    with col2:
        prompt = st.text_input(tr("Describe a drawing you want (e.g., 'cute baby lion with a crown')",
                                 "صف رسمة تريدها (مثلاً: 'أسد صغير لطيف مع تاج')"))
        if st.button(tr("🎨 Generate Cute Drawing (Stability)","🎨 أنشئ رسمة لطيفة (Stability)")):
            img_bytes = generate_drawing_with_stability(prompt)
            if img_bytes: st.image(img_bytes, caption=tr("Generated Art","صورة مولّدة"), use_column_width=True)
            else: st.warning(tr("Couldn't generate drawing (check STABILITY_API_KEY in your .env).",
                               "تعذّر توليد الرسمة (تحقّق من STABILITY_API_KEY في ملف .env)."))

elif tab == tr("🛠️ Dad's Dashboard", "🛠️ لوحة تحكم الأب"):
    st.title(tr("🛠️ Dad's Dashboard", "🛠️ لوحة تحكم الأب"))
    if st.button(tr("📧 Send Test Email to Dad","📧 أرسل رسالة تجريبية للأب")):
        with st.spinner(tr("Sending test email...","جاري إرسال رسالة تجريبية...")):
            ok, msg = send_email_to_dad(tr("Test Kid","طفل اختبار"),
                                        tr("This is a test email.","هذه رسالة تجريبية."),
                                        tr("This is a test answer.","هذه إجابة تجريبية."))
        if ok: st.success(tr("Test email sent ✅","تم إرسال الرسالة ✅")); st.caption(msg)
        else: st.error(tr("Test email failed ❌","فشل إرسال الرسالة ❌")); st.code(msg, language="text")
    render_dashboard_tab()

elif tab == tr("📚 Learning Book", "📚 كتاب التعلم"):
    st.title(tr("📚 Learning Book","📚 كتاب التعلم"))
    render_learning_book_tab_local()

elif tab == tr("🧠 Quiz Fun", "🧠 مسابقة ممتعة"):
    st.title(tr("🧠 Quiz Fun","🧠 مسابقة ممتعة"))
    st.write(tr("Answer fun questions and get stars!","أجب عن أسئلة ممتعة واحصل على نجوم!"))
    name = st.text_input(tr("Your name for the scoreboard:","اسمك في لوحة النتائج:"), key="quiz_name")
    if "quiz_started" not in st.session_state: st.session_state.quiz_started = False
    if "quiz_score" not in st.session_state: st.session_state.quiz_score = 0
    if "quiz_q_index" not in st.session_state: st.session_state.quiz_q_index = 0
    if not st.session_state.quiz_started:
        if st.button(tr("▶️ Start Quiz","▶️ ابدأ المسابقة")):
            st.session_state.quiz_started = True; st.rerun()
    else:
        q = get_quiz_question(st.session_state.quiz_q_index)
        if q is None:
            st.subheader(tr("🏁 Finished! Your score:","🏁 انتهيت! نتيجتك:") + f" {st.session_state.quiz_score}/5")
            stars = "⭐"*st.session_state.quiz_score + "☆"*(5 - st.session_state.quiz_score)
            st.markdown(f"### {stars}")
            play_win_sound()
            if name.strip(): log_score(name.strip(), st.session_state.quiz_score)
            if st.button(tr("🔁 Play Again","🔁 العب مجدداً")):
                for key in list(st.session_state.keys()):
                    if key.startswith("quiz_"): del st.session_state[key]
                st.rerun()
        else:
            st.subheader(q["question"])
            # Colorful alphabet-style radio labels
            labeled = [f"{chr(65+i)}. {opt}" for i, opt in enumerate(q["choices"])]
            choice = st.radio(tr("Pick one:","اختر واحدة:"), labeled, key=f"quiz_choice_{st.session_state.quiz_q_index}")
            if st.button(tr("✅ Submit","✅ أرسل")):
                idx = labeled.index(choice)
                picked = q["choices"][idx]
                if picked == q["answer"]:
                    st.success(tr("Correct! 🎉","صحيح! 🎉")); play_correct_sound(); st.session_state.quiz_score += 1
                else:
                    st.error(tr("Oops! Correct answer is:","عفواً! الجواب الصحيح هو:") + f" {q['answer']}")
                    play_wrong_sound()
                st.session_state.quiz_q_index += 1; st.rerun()

elif tab == tr("📊 Scoreboard", "📊 لوحة النتائج"):
    st.title(tr("📊 Quiz Scoreboard","📊 لوحة النتائج"))
    show_scoreboard()

elif tab == tr("🎨 Draw & Guess (Gemini)", "🎨 ارسم وخمّن (Gemini)"):
    st.title(tr("🎨 Draw & Guess (Gemini)","🎨 ارسم وخمّن (Gemini)"))
    stroke_w = st.slider(tr("Pen size","حجم القلم"), 4, 30, 12); bg = st.color_picker(tr("Background","الخلفية"), "#FFFFFF")
    st.write(tr("Tip: use black pen on white background for best results.",
               "نصيحة: استخدم قلماً أسود وخلفية بيضاء لنتائج أفضل."))
    canvas_result = st_canvas(fill_color="rgba(0, 0, 0, 0)", stroke_width=stroke_w, stroke_color="#000000",
                              background_color=bg, update_streamlit=True, height=300, width=300,
                              drawing_mode="freedraw", key="canvas")
    col1, col2 = st.columns(2); guess = None
    if col1.button(tr("🤖 Guess with Gemini","🤖 تخمين باستخدام Gemini")):
        if canvas_result.image_data is not None:
            from PIL import Image
            img = Image.fromarray(canvas_result.image_data.astype("uint8"), "RGBA")
            img = img.convert("L").point(lambda x: 0 if x < 250 else 255).convert("RGB")
            buf = BytesIO(); img.save(buf, format="PNG"); png_bytes = buf.getvalue()
            with st.spinner(tr("Asking Gemini...","سؤال Gemini...")):
                try:
                    guess = classify_sketch(png_bytes)
                except Exception as e:
                    st.error(f"Gemini error: {e}"); guess = None
        else:
            st.info(tr("Please draw something first.","من فضلك ارسم شيئاً أولاً."))
    if guess:
        st.success(tr(
            f"I think it's a **{guess.get('animal','unknown')}** (certainty {guess.get('certainty',0):.2f})",
            f"أعتقد أنها **{guess.get('animal','غير معروف')}** (درجة الثقة {guess.get('certainty',0):.2f})"
        ))
        if guess.get("alternatives"): st.caption(tr("Other ideas: ","أفكار أخرى: ") + ", ".join(guess["alternatives"]))
        if col2.button(tr("📷 Show real photo","📷 عرض صورة حقيقية")):
            with st.spinner(tr("Finding a photo...","جاري العثور على صورة...")):
                url = fetch_animal_photo(guess.get("animal",""))
            if url: st.image(url, caption=tr("Real photo","صورة حقيقية"), use_column_width=True)
            else: st.warning(tr("Couldn't find a photo right now. Try another animal or check your internet.",
                               "تعذّر العثور على صورة الآن. جرّب حيواناً آخر أو تحقّق من الإنترنت."))
