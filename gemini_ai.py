import os
import io
import json
import requests
from PIL import Image
import google.generativeai as genai

# Configure Gemini with API key from environment
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# Choose a fast, vision-capable model
_MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

def _get_model():
    try:
        return genai.GenerativeModel(_MODEL_NAME)
    except Exception as e:
        raise RuntimeError(f"Failed to initialize Gemini model '{_MODEL_NAME}': {e}")

# ---------- Text Q&A ----------
def ask_gemini(question: str) -> str:
    """Ask Gemini for a short, kid-friendly answer."""
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY not set in environment")
    model = _get_model()
    prompt = (
        "You are a friendly assistant for kids. "
        "Answer simply and briefly.\n\n"
        f"Question: {question}"
    )
    resp = model.generate_content(prompt)
    return (resp.text or "").strip() if resp else "Sorry, I couldn't answer right now."

# ---------- Sketch classification ----------
def classify_sketch(image_bytes: bytes) -> dict:
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY not set in environment")
    model = _get_model()
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    system_prompt = (
        "You are a friendly kids app that guesses animals from simple black-and-white sketches. "
        "Look only at the drawing. Guess the single best animal name (e.g., 'cat', 'dog', 'lion', 'elephant'). "
        "If it's not an animal, answer 'unknown'. "
        "Respond in strict JSON with keys: animal (string, lowercase), alternatives (array of 2-4 strings), certainty (0-1 float). "
        "Example: {\"animal\":\"cat\",\"alternatives\":[\"tiger cub\",\"kitten\"],\"certainty\":0.82}"
    )
    resp = model.generate_content([system_prompt, img])
    text = (resp.text or "").strip()
    try:
        guess = json.loads(text); guess["raw"] = text; return guess
    except Exception:
        import re
        m = re.search(r"\{[\s\S]*\}", text)
        if m:
            try:
                guess = json.loads(m.group(0)); guess["raw"] = text; return guess
            except Exception:
                pass
        return {"animal":"unknown","alternatives":[],"certainty":0.0,"raw":text}

# ---------- Real photo via Wikipedia ----------
def fetch_animal_photo(animal: str) -> str | None:
    if not animal or animal == "unknown":
        return None
    title = animal.strip().replace(" ", "_").capitalize()
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{title}"
    try:
        r = requests.get(url, timeout=8)
        if r.status_code == 200:
            data = r.json()
            thumb = data.get("thumbnail",{}).get("source")
            if thumb:
                return thumb
            orig = data.get("originalimage",{}).get("source")
            if orig:
                return orig
        return None
    except Exception:
        return None
