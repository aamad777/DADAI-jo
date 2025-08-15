# book_store.py — simple local book library with TF‑IDF retrieval
import os, json, uuid, re
from typing import Dict, List, Tuple

LIB_DIR = "books"
ACTIVE_FILE = os.path.join(LIB_DIR, "_active.json")
os.makedirs(LIB_DIR, exist_ok=True)

def _clean(s: str) -> str:
    s = re.sub(r"\s+", " ", s or "").strip()
    return s

def chunk_text(text: str, chunk_size: int = 900, overlap: int = 150) -> List[str]:
    text = _clean(text)
    chunks, i = [], 0
    while i < len(text):
        chunks.append(text[i:i+chunk_size])
        i += max(1, chunk_size - overlap)
    return chunks

def _book_path(book_id: str) -> str:
    return os.path.join(LIB_DIR, f"{book_id}.json")

def save_book(title: str, lang: str, full_text: str) -> str:
    book_id = str(uuid.uuid4())[:8]
    chunks = chunk_text(full_text)
    payload = {
        "id": book_id,
        "title": _clean(title) or f"Book {book_id}",
        "lang": lang,
        "chunks": chunks,
        "size": sum(len(c) for c in chunks),
    }
    with open(_book_path(book_id), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return book_id

def load_book(book_id: str) -> Dict:
    try:
        with open(_book_path(book_id), "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def delete_book(book_id: str) -> None:
    try:
        os.remove(_book_path(book_id))
    except Exception:
        pass
    active = get_active_book_id()
    if active == book_id:
        set_active_book_id(None)

def list_books() -> List[Dict]:
    out = []
    for fn in os.listdir(LIB_DIR):
        if not fn.endswith(".json") or fn == "_active.json":
            continue
        try:
            with open(os.path.join(LIB_DIR, fn), "r", encoding="utf-8") as f:
                meta = json.load(f)
                out.append({"id": meta["id"], "title": meta["title"], "lang": meta.get("lang",""), "num_chunks": len(meta.get("chunks",[]))})
        except Exception:
            continue
    out.sort(key=lambda x: x["title"].lower())
    return out

def set_active_book_id(book_id: str | None):
    with open(ACTIVE_FILE, "w", encoding="utf-8") as f:
        json.dump({"active_id": book_id}, f)

def get_active_book_id() -> str | None:
    try:
        with open(ACTIVE_FILE, "r", encoding="utf-8") as f:
            return (json.load(f) or {}).get("active_id")
    except Exception:
        return None

def find_relevant_chunks(query: str, chunks: List[str], top_k: int = 4) -> List[Tuple[str, float]]:
    if not chunks:
        return []
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        vect = TfidfVectorizer(analyzer="char", ngram_range=(3,5), min_df=1)
        X = vect.fit_transform(chunks + [query])
        sims = cosine_similarity(X[-1], X[:-1]).flatten()
        idx = sims.argsort()[::-1][:top_k]
        return [(chunks[i], float(sims[i])) for i in idx]
    except Exception:
        q = query.lower()
        scored = []
        for ch in chunks:
            score = 0.0
            for w in set(q.split()):
                score += ch.lower().count(w)
            scored.append((ch, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

def build_context_for_question(question: str, book_id: str, top_k: int = 4) -> Tuple[str, float]:
    meta = load_book(book_id)
    chunks = meta.get("chunks", [])
    pairs = find_relevant_chunks(question, chunks, top_k=top_k)
    ctx = "\n\n".join(p[0] for p in pairs)
    best = max([p[1] for p in pairs], default=0.0)
    return ctx, best
