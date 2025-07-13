import streamlit as st
import google.generativeai as genai
import speech_recognition as sr
import os
import tempfile

# --- PAGE CONFIG ---
st.set_page_config(page_title="Gemini Chatbot", layout="wide")
st.title("🤖 Gemini + ChatGPT Assistant")

# --- SIDEBAR SETTINGS ---
with st.sidebar:
    st.header("🔐 API Configuration")
    api_key = st.text_input("Enter your Gemini API Key", type="password")
    model_choice = st.selectbox("Select Model", ["gemini-1.5-flash", "gemini-1.5-pro"])
    st.markdown("""
        **Note:** Get your key from [Google AI Studio](https://makersuite.google.com/app)
    """)

# --- INITIALIZATION ---
if api_key:
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_choice)
        st.success(f"Model '{model_choice}' loaded successfully.")
    except Exception as e:
        st.error(f"Error initializing Gemini: {e}")
        st.stop()
else:
    st.warning("Please enter your Gemini API key to continue.")
    st.stop()

# --- SESSION STATE ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- DISPLAY CHAT HISTORY ---
for entry in st.session_state.chat_history:
    with st.chat_message(entry["role"]):
        st.markdown(entry["text"])

# --- FILE UPLOAD ---
st.subheader("📁 Upload File (optional)")
uploaded_file = st.file_uploader("Upload any file to include in your prompt.", type=None)
file_reference = None
if uploaded_file:
    try:
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name
        file_reference = genai.upload_file(path=tmp_path)
        st.success(f"Uploaded file: {uploaded_file.name}")
    except Exception as e:
        st.error(f"File upload error: {e}")

# --- VOICE INPUT ---
st.subheader("🎙️ Voice Input")
use_voice = st.toggle("Use microphone input")
voice_prompt = ""
if use_voice and st.button("Record Voice"):
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        with st.spinner("Listening..."):
            try:
                recognizer.adjust_for_ambient_noise(source)
                audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
                voice_prompt = recognizer.recognize_google(audio)
                st.success("Voice recognized!")
                st.text_input("Voice Transcription", value=voice_prompt, key="voice_text")
            except Exception as e:
                st.error(f"Voice recognition error: {e}")

# --- TEXT INPUT ---
prompt = voice_prompt if voice_prompt else st.chat_input("Type your message")

if prompt:
    st.session_state.chat_history.append({"role": "user", "text": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    try:
        inputs = [file_reference, prompt] if file_reference else [prompt]
        response = model.generate_content(inputs)
        reply = response.text
        st.session_state.chat_history.append({"role": "assistant", "text": reply})
        with st.chat_message("assistant"):
            st.markdown(reply)
    except Exception as e:
        st.error(f"Response error: {e}")

# --- DOWNLOAD CHAT HISTORY ---
st.sidebar.markdown("---")
if st.sidebar.button("📄 Export Chat History"):
    chat_text = "\n\n".join([f"{msg['role'].capitalize()}: {msg['text']}" for msg in st.session_state.chat_history])
    st.sidebar.download_button("Download .txt", chat_text, file_name="gemini_chat.txt")
