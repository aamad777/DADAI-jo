# drawing.py — Stability image generation (fallback to None if key missing)

import os
import base64
import requests

def generate_drawing_with_stability(prompt: str):
    """
    Returns raw image bytes if STABILITY_API_KEY is set and the request succeeds, else None.
    """
    api_key = os.getenv("STABILITY_API_KEY", "")
    if not api_key or not prompt.strip():
        return None
    try:
        # Using Stability "SD3" style prompt (endpoint may vary on your plan)
        url = "https://api.stability.ai/v2beta/stable-image/generate/ultra"
        headers = {"Authorization": f"Bearer {api_key}"}
        data = {"prompt": prompt, "output_format": "png"}
        r = requests.post(url, headers=headers, files={"none": ""}, data=data, timeout=60)
        r.raise_for_status()
        # API returns raw bytes or base64 depending on plan; handle both
        if r.headers.get("content-type","").startswith("image/"):
            return r.content
        js = r.json()
        if "image" in js:
            return base64.b64decode(js["image"])
    except Exception:
        pass
    return None
