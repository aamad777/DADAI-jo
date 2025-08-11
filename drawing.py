import os
import requests
import base64
from dotenv import load_dotenv

load_dotenv()

def generate_drawing_with_stability(prompt):
    stability_api_key = os.getenv("STABILITY_API_KEY")
    if not stability_api_key:
        return None

    try:
        response = requests.post(
            "https://api.stability.ai/v1/generation/stable-diffusion-xl-1024-v1-0/text-to-image",
            headers={
                "Authorization": f"Bearer {stability_api_key}",
                "Accept": "application/json",
                "Content-Type": "application/json",
            },
            json={
                "text_prompts": [{"text": prompt}],
                "cfg_scale": 7,
                "height": 896,
                "width": 1152,
                "samples": 1,
                "steps": 30
            },
        )

        if response.status_code == 200:
            data = response.json()
            base64_image = data["artifacts"][0]["base64"]
            return base64.b64decode(base64_image)
        else:
            print("Stability API error:", response.status_code, response.text)
            return None

    except Exception as e:
        print("Drawing error:", e)
        return None
