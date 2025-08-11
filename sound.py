import os

def play_animal_sound(animal):
    sound_path = f"static/sounds/{animal.lower()}.mp3"
    if os.path.exists(sound_path):
        with open(sound_path, "rb") as audio_file:
            return audio_file.read()
    return None
