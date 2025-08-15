# rag_arabic.py — Arabic-friendly PDF extraction + simple RAG index

import re, os, tempfile
from typing import List, Tuple

# ---------- Arabic normalization ----------
ARABIC_DIACRITICS = re.compile(r"[\u0610-\u061A\u064B-\u065F\u06D6-\u06ED]")

def normalize_arabic(s: str) -> str:
    s = ARABIC_DIACRITICS.sub("", s)
    s = s.replace("ـ", "")            # tatweel
    s = re.sub("[إأآا]", "ا", s)      # alifs
    s = s.replace("ى", "ي").replace("ئ", "ي").replace("ؤ", "و").replace("ة", "ه")
    s = re.sub(r"\s+", " ", s).strip()
    return s

# ---------- Digital text (PyMuPDF blocks) ----------
def extract_text_pymupdf_blocks(pdf_bytes: bytes) -> str:
    try:
        import fitz  # PyMuPDF
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

# ---------- Fallback: pdfminer ----------
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

# ---------- OCR: ocrmypdf (best) ----------
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

# ---------- OCR: pytesseract + pdf2image ----------
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

# ---------- Master extractor ----------
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

# ---------- Chunking ----------
def chunk_text(text: str, chunk_chars: int = 1200, overlap: int = 250) -> List[str]:
    import re
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

# ---------- Local embedding index (multilingual) ----------
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

        import faiss, numpy as np
        d = embs.shape[1]
        self.index = faiss.IndexFlatIP(d)
        self.index.add(embs.astype("float32"))

    def search_text(self, query: str, top_k: int = 5) -> str:
        if self.index is None:
            return ""
        self._load_model()
        import numpy as np, faiss
        qv = self.model.encode([normalize_arabic(query)], normalize_embeddings=True).astype("float32")
        D, I = self.index.search(qv, top_k)
        parts = []
        for idx in I[0]:
            if idx >= 0:
                parts.append(self.chunks[int(idx)])
        return "\n\n".join(parts)
