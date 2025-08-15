# sound.py — simple placeholder that won't crash if you don't have local sound files
import streamlit as st

def play_animal_sound(animal: str):
    # If you have local wavs, you can map and st.audio them.
    # This is a safe placeholder:
    st.success(f"🐾 (Pretend) Playing {animal} sound!")
