# app.py — Ask Dad AI (name + age with colorful alphabet & numbers)
# ---------------------------------------------------------------
# Fixes:
#  - removes undefined "_" usage (UnboundLocalError) and adds a tiny tr() i18n helper
#  - avoids writing to st.session_state after widget creation (StreamlitAPIException)
#  - uses one safe key per widget and initializes state BEFORE widgets are created
# Features:
#  - Colorful A–Z alphabet and digits 1–9
#  - Kid's name & age card; age styled EXACTLY like alphabet letters

import streamlit as st
import string
from itertools import cycle

st.set_page_config(page_title="Ask DAD AI", page_icon="👨‍👧", layout="centered")

# ------------------ i18n helper (replaces the undefined `_`) ------------------
def tr(en: str, ar: str | None = None) -> str:
    """Return UI text based on session language; defaults to English."""
    lang = (st.session_state.get("ui_lang") or "en").lower()
    if lang.startswith("ar") and ar:
        return ar
    return en

# ----------------------- SAFE state init (before widgets) ----------------------
if "ui_lang" not in st.session_state:
    st.session_state["ui_lang"] = "en"  # change to "ar" if you want Arabic by default

if "child_name" not in st.session_state:
    st.session_state["child_name"] = tr("Kid", "طفل")

if "child_age" not in st.session_state:
    st.session_state["child_age"] = 4  # 1–9

# ------------------------------- UI Header ------------------------------------
st.markdown(
    f"""
    <div style="display:flex;align-items:center;gap:14px;margin:6px 0 2px 0;">
      <div style="font-size:34px">👨‍👧</div>
      <div style="font-weight:800;font-size:42px;letter-spacing:.5px">Ask DAD AI</div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.caption(tr("Colorful alphabet and numbers for your kid's name & age",
              "أبجدية وأرقام ملوّنة لاسم طفلك وعمره"))

# ----------------------------- Input widgets ----------------------------------
# NOTE: We do not mutate these keys after creation; the widgets own them.
name = st.text_input(
    tr("Kid's name", "اسم الطفل"),
    key="child_name",
    value=st.session_state["child_name"],
    placeholder=tr("Type the name…", "أدخل الاسم…"),
)

age = st.slider(tr("Age (1–9)", "العمر (١–٩)"), 1, 9, key="child_age")

st.divider()

# ------------------------- Color rendering helpers ----------------------------
PALETTE = [
    "#ef4444", "#f59e0b", "#10b981", "#3b82f6", "#8b5cf6",
    "#ec4899", "#14b8a6", "#eab308", "#22c55e", "#0ea5e9",
    "#a855f7", "#f43f5e"
]

def colored_row(chars, size_px: int = 42, gap_px: int = 8, weight: int = 900) -> str:
    """
    Return HTML that renders a sequence of characters, each with a cycling color.
    Used for: alphabet, digits, name, and age — SAME style everywhere.
    """
    cyc = cycle(PALETTE)
    spans = []
    for ch in chars:
        color = next(cyc)
        # keep spaces visible but uncolored
        if ch.strip() == "":
            spans.append(f'<span style="display:inline-block;width:{gap_px}px"></span>')
            continue
        spans.append(
            f'<span style="display:inline-block;margin-right:{gap_px}px;'
            f'font-weight:{weight};font-size:{size_px}px;color:{color};">{ch}</span>'
        )
    return "".join(spans)

# -------------------------- Alphabet (A–Z) ------------------------------------
st.markdown("**" + tr("Colorful Alphabet", "أبجدية ملوّنة") + "**")
alphabet_html = colored_row(list(string.ascii_uppercase), size_px=42, gap_px=8, weight=900)
st.markdown(f'<div style="line-height:2">{alphabet_html}</div>', unsafe_allow_html=True)

# -------------------------- Numbers (1–9) -------------------------------------
# EXACT same style as alphabet (same function, same params)
st.markdown("**" + tr("Numbers (1–9)", "الأرقام (١–٩)") + "**")
digits_html = colored_row([str(i) for i in range(1, 10)], size_px=42, gap_px=8, weight=900)
st.markdown(f'<div style="line-height:2">{digits_html}</div>', unsafe_allow_html=True)

st.divider()

# ----------------------- Kid's colorful name & age card -----------------------
display_name = (name or "").upper()
display_age = str(age)

name_html = colored_row(list(display_name), size_px=56, gap_px=10, weight=900)
age_html  = colored_row(list(display_age), size_px=56, gap_px=10, weight=900)  # SAME STYLE

st.markdown("### " + tr("Your Colorful Card", "بطاقة ملوّنة"))
st.markdown(
    f"""
    <div style="
        padding:18px;border:1px solid #eee;border-radius:16px;
        box-shadow:0 1px 8px rgba(0,0,0,.05);">
      <div style="margin-bottom:8px;color:#64748b;font-weight:700;">{tr("Name","الاسم")}</div>
      <div style="margin-bottom:12px;">{name_html}</div>
      <div style="margin:8px 0;color:#64748b;font-weight:700;">{tr("Age","العمر")}</div>
      <div>{age_html}</div>
    </div>
    """,
    unsafe_allow_html=True
)

# ----------------------------- Notes / Tips -----------------------------------
with st.expander(tr("Tips", "نصائح")):
    st.write(
        tr(
            "- The alphabet, numbers, name, and age all use the exact same style function.\n"
            "- To switch UI language, set `st.session_state['ui_lang'] = 'ar'`.",
            "- الأبجدية والأرقام والاسم والعمر تستخدم نفس نمط الألوان بالضبط.\n"
            "- لتبديل اللغة، عدّل `st.session_state['ui_lang'] = 'ar'`."
        )
    )
