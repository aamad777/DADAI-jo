# gemini_ai.py — thin wrappers (Gemini → OpenAI → stub)
import os, random
from typing import Dict, Any, Optional
USE_GEMINI = False
try:
    import google.generativeai as genai
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
    if GEMINI_API_KEY:
        genai.configure(api_key=GEMINI_API_KEY)
        USE_GEMINI = True
except Exception:
    USE_GEMINI = False

from openai import OpenAI
_openai_key = os.getenv("OPENAI_API_KEY", "")
_openai = OpenAI(api_key=_openai_key) if _openai_key else None

def ask_gemini(prompt: str) -> str:
    if USE_GEMINI:
        model = genai.GenerativeModel("gemini-1.5-flash")
        resp = model.generate_content(prompt)
        return (resp.text or "").strip()
    if _openai:
        resp = _openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5, max_tokens=300
        )
        return (resp.choices[0].message.content or "").strip()
    return "I couldn't answer right now."

def classify_sketch(png_bytes: bytes) -> Dict[str, Any]:
    if USE_GEMINI:
        model = genai.GenerativeModel("gemini-1.5-flash")
        resp = model.generate_content(
            [{"mime_type": "image/png", "data": png_bytes}, "What's the animal sketch? Give best guess + 2 alternatives and confidence 0-1."]
        )
        txt = (resp.text or "").strip()
        return {"animal": txt.splitlines()[0][:20], "alternatives": [], "certainty": 0.66}
    animals = ["cat", "dog", "lion", "elephant", "fish", "bird"]
    pick = random.choice(animals)
    return {"animal": pick, "alternatives": random.sample(animals, 2), "certainty": round(random.uniform(0.5, 0.9), 2)}

def fetch_animal_photo(animal: str) -> Optional[str]:
    pics = {
        "cat": "https://images.unsplash.com/photo-1592194996308-7b43878e84a2?q=80&w=1080&auto=format&fit=crop",
        "dog": "https://images.unsplash.com/photo-1518020382113-a7e8fc38eac9?q=80&w=1080&auto=format&fit=crop",
        "lion": "https://images.unsplash.com/photo-1603387863769-96835b0fb2ad?q=80&w=1080&auto=format&fit=crop",
        "elephant": "https://images.unsplash.com/photo-1507149833265-60c372daea22?q=80&w=1080&auto=format&fit=crop",
        "fish": "https://images.unsplash.com/photo-1518837695005-2083093ee35b?q=80&w=1080&auto=format&fit=crop",
        "bird": "https://images.unsplash.com/photo-1470167290877-7d5d3446de4c?q=80&w=1080&auto=format&fit=crop",
    }
    key = (animal or "").strip().lower()
    return pics.get(key)
