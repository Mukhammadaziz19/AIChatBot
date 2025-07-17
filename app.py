import streamlit as st
import google.generativeai as genai
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase
import tempfile
import numpy as np
import av
import whisper
import os

# --- PAGE CONFIG ---
st.set_page_config(page_title="Gemini Voice Assistant", layout="wide")
st.title("🤖 Gemini AI + Voice Chatbot")

# --- SIDEBAR SETTINGS ---
with st.sidebar:
    st.header("🔐 API Configuration")
    api_key = st.text_input("Enter your Gemini API Key", type="password")
    model_choice = st.selectbox("Select Model", ["gemini-1.5-flash", "gemini-1.5-pro"])
    st.markdown("Get your key from [Google AI Studio](https://makersuite.google.com/app)")

# --- INITIALIZATION ---
if not api_key:
    st.warning("Please enter your Gemini API key to continue.")
    st.stop()

try:
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_choice)
    st.success(f"Model '{model_choice}' loaded successfully.")
except Exception as e:
    st.error(f"Gemini Init Error: {e}")
    st.stop()

# --- SESSION STATE ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- FILE UPLOAD ---
st.subheader("📁 Upload File (optional)")
uploaded_file = st.file_uploader("Upload any file for context", type=None)
file_reference = None
if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name
    try:
        file_reference = genai.upload_file(path=tmp_path)
        st.success(f"Uploaded file: {uploaded_file.name}")
    except Exception as e:
        st.error(f"File upload error: {e}")

# --- VOICE INPUT ---
st.subheader("🎙️ Voice Input")
voice_prompt = ""

class AudioProcessor(AudioProcessorBase):
    def __init__(self):
        self.buffer = []

    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        audio = frame.to_ndarray().flatten().astype(np.float32)
        self.buffer.extend(audio.tolist())
        return frame

    def get_audio(self):
        return np.array(self.buffer)

use_voice = st.toggle("Use microphone input")

if use_voice:
    ctx = webrtc_streamer(
        key="voice-chat",
        audio_processor_factory=AudioProcessor,
        media_stream_constraints={"audio": True, "video": False},
        async_processing=True,
    )

    if ctx.audio_processor and st.button("🎤 Transcribe Voice"):
        audio_data = ctx.audio_processor.get_audio()
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(audio_data.tobytes())
            f.flush()
            try:
                st.info("Transcribing...")
                whisper_model = whisper.load_model("base")
                result = whisper_model.transcribe(f.name)
                voice_prompt = result["text"]
                st.success("Voice recognized!")
                st.text_input("Voice Transcription", value=voice_prompt, key="voice_text")
            except Exception as e:
                st.error(f"Transcription failed: {e}")

# --- DISPLAY CHAT HISTORY ---
for entry in st.session_state.chat_history:
    with st.chat_message(entry["role"]):
        st.markdown(entry["text"])

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
        st.error(f"Gemini Error: {e}")

# --- DOWNLOAD CHAT HISTORY ---
st.sidebar.markdown("---")
if st.sidebar.button("📄 Export Chat History"):
    chat_text = "\n\n".join([f"{msg['role'].capitalize()}: {msg['text']}" for msg in st.session_state.chat_history])
    st.sidebar.download_button("Download .txt", chat_text, file_name="gemini_chat.txt")
