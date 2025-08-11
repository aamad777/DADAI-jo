
import streamlit as st

def play_correct_sound():
    st.audio("https://www.soundjay.com/button/beep-07.wav")

def play_wrong_sound():
    st.audio("https://www.soundjay.com/button/beep-10.wav")

def play_win_sound():
    st.audio("https://www.soundjay.com/human/cheering-01.mp3")
