import streamlit as st
import sounddevice as sd
import numpy as np
import wave
from transformers import pipeline
import whisper

# Set page configuration
st.set_page_config(
    page_title="Whisper App",
    page_icon="🔊",
    layout="wide",
    initial_sidebar_state="expanded",  # Set sidebar to be expanded by default
)

# App title and description
st.title("Whisper App")
st.markdown(
    "This app transcribes uploaded audio files and generates key points from the transcription."
)

# Upload audio file with streamlit
audio_file = st.file_uploader(
    "Upload Audio", type=["wav", "mp3", "m4a"], key="audio_uploader"
)

model = whisper.load_model("base")
st.text("Whisper Model Loaded")

# Initialize temp_wav_file outside of the recording block
temp_wav_file = "temp_audio.wav"

# Record audio using the microphone
recording = st.checkbox("Record Audio")
st.markdown(
    "You Can record only for 20seconds"
)


if recording:
    audio_file = None
    fs = 44100  # Sample rate
    duration = 20  # Recording duration in seconds
    st.info("Recording...")
    st.info("This app take time to load so plese start recording when you get a microphone icon in the right side of your taskbar  ")

    

    audio_data = sd.rec(int(fs * duration), samplerate=fs, channels=1, dtype="int16")
    sd.wait()
    with wave.open(temp_wav_file, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(fs)
        wf.writeframes(audio_data.tobytes())
    st.audio(temp_wav_file, format="audio/wav")
    st.success("Recording complete! Click 'Transcribe Audio' to process.")

# Transcribe audio
if st.sidebar.button("Transcribe Audio", key="transcribe_btn"):
    with st.spinner("Transcribing Audio..."):
        if audio_file is not None:
            transcription = whisper.load_model("base").transcribe(audio_file.name)
        elif recording:            
            transcription = whisper.load_model("base").transcribe(temp_wav_file)
        else:
            st.sidebar.error("Please upload an audio file or record audio")

    st.success("Transcription Complete")
    st.markdown(transcription["text"])

    # Store the transcription result in the session state variable
    st.session_state.transcription_result = transcription["text"]

# Generate Key Points section
if st.sidebar.button("Generate Key Points"):
    if "transcription_result" in st.session_state and st.session_state.transcription_result:
        summarizer = pipeline("summarization")
        key_points = summarizer(
            st.session_state.transcription_result, max_length=150, min_length=10, do_sample=False
        )

            # Display key points as bullet points
        st.sidebar.success("Key Points Generated")
        key_points_list = key_points[0]["summary_text"].split(". ")
        for point in key_points_list:
            st.sidebar.markdown(f"- {point}")

            # Display summary in the main area
        st.success("Summary")
        st.markdown(key_points[0]["summary_text"])
    else:
        st.sidebar.warning("Please transcribe audio before generating key points.")


# Improved styling and layout
st.sidebar.header("Audio File")
if audio_file is not None:
    st.sidebar.audio(audio_file, format="audio/wav")
elif recording:
    st.sidebar.audio(temp_wav_file, format="audio/wav")

# Additional features and explanations
st.sidebar.header("Instructions")
st.sidebar.markdown(
    "1. Upload an audio file or record using the microphone.\n"
    "2. Click 'Transcribe Audio' to get the transcription.\n"
    "3. Use 'Generate Key Points' to summarize the transcription."
)

# Disclaimer
st.sidebar.info(
    "Note: This is a simplified example and may not cover all edge cases or considerations."
)
