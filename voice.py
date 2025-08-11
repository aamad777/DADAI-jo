# voice_io.py
from io import BytesIO

def tts_gtts_bytes(text: str, lang: str = "en", slow: bool = False) -> bytes:
    """
    Generate speech (MP3 bytes) from text using gTTS.
    Usage:
        mp3_bytes = tts_gtts_bytes("Hello!", lang="en")
        # In Streamlit: st.audio(mp3_bytes, format="audio/mp3")
    """
    if not text:
        return b""

    try:
        from gtts import gTTS
    except ImportError as e:
        raise ImportError("gTTS is not installed. Run: pip install gTTS") from e

    mp3_fp = BytesIO()
    tts = gTTS(text=text, lang=lang, slow=slow)
    tts.write_to_fp(mp3_fp)
    return mp3_fp.getvalue()
