import streamlit as st
import os
import pyaudio
import wave
import whisper
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import difflib

# === Snellen Chart Reference ===
snellen_chart = {
    1: "E",
    2: "F P",
    3: "T O Z",
    4: "L P E D",
    5: "P E C F D",
    6: "E D F C Z P",
    7: "F E L O P Z D",
    8: "D E F P O T E C"
}

transcriptionlcs=""
# === Audio Settings ===
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024
OUTPUT_FILENAME = "Recorded_audio.wav"

final=0
# === Function: Analyze Transcription ===
def analyze_transcription(transcription):
    # transcribed_words = transcription.split()
    transcribed_words=[]
    for i in transcription:
        if i not in [' ',',','.']:
            transcribed_words.append(i)
    st.code(transcribed_words)
    global transcriptionlcs
    transcriptionlcs=''.join(transcribed_words)
    st.code(transcriptionlcs)

    total_correct = 0
    total_letters = 0

    # lower_lines = [6, 7, 8]
    # upper_lines = [1, 2, 3, 4, 5]

    # lower_accuracy = []
    # upper_accuracy = []

    for line_number, expected_text in snellen_chart.items():
        expected_words = expected_text.split()
        recognized_words = transcribed_words[:len(expected_words)]
        transcribed_words = transcribed_words[len(expected_words):]

        line_correct = sum(e == r for e, r in zip(expected_words, recognized_words))
        total_correct += line_correct
        total_letters += len(expected_words)

        line_accuracy = line_correct / len(expected_words) * 100 if expected_words else 0

        # if line_number in lower_lines:
        #     lower_accuracy.append(line_accuracy)
        # elif line_number in upper_lines:
        #     upper_accuracy.append(line_accuracy)

        with st.expander(f"Line {line_number}: Accuracy {line_accuracy:.2f}%"):
            st.markdown(f"*Expected:* {expected_text}")
            st.markdown(f"*Recognized:* { ' '.join(recognized_words) if recognized_words else '---' }")
            for e, r in zip(expected_words, recognized_words):
                status = "✅" if e == r else "❌"
                st.markdown(f"{status} *Expected:* {e} | *Heard:* {r}")

        if not transcribed_words:
            
            break

    final_accuracy = total_correct / total_letters * 100 if total_letters > 0 else 0
    # # Vision Diagnosis
    # avg_lower = np.sum(lower_accuracy)/3 if lower_accuracy else 0
    # avg_upper = np.sum(upper_accuracy)/5 if upper_accuracy else 0

    # st.subheader("🧠 Vision Analysis Based on Accuracy")
    # st.markdown(f"🔻 *Average Accuracy (Lines 6–8):* {avg_lower:.2f}%")
    # st.markdown(f"🔺 *Average Accuracy (Lines 1–5):* {avg_upper:.2f}%")

    # if avg_lower <=  60:
    #     st.error("🩺 Likely *Myopia* (Difficulty with far vision) For treatment Contact a Ophthalmologist")
    #     flag=0

    # if avg_upper <= 60:
    #     st.error("🩺 Likely *Hypermetropia* (Difficulty with near vision) For treatment Contact a Ophthalmologist")
    #     flag=0

    if final_accuracy<=65:
        st.error("🩺 Medical Supervision needed. For treatment Contact a Ophthalmologist")

    else:
        st.success("🩺 *No significant vision issues detected* based on this test")
    global final
    final=final_accuracy
    return final_accuracy

# === Function: Audio File Loading ===
def load_audio(file_path):
    y, sr = librosa.load(file_path, sr=None)
    return y, sr

def compute_spectrogram(y, sr, n_fft=2048, hop_length=512):
    D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    return S_db

def pad_spectrogram(S1, S2):
    max_len = max(S1.shape[1], S2.shape[1])
    S1_padded = np.pad(S1, ((0, 0), (0, max_len - S1.shape[1])), mode='constant')
    S2_padded = np.pad(S2, ((0, 0), (0, max_len - S2.shape[1])), mode='constant')
    return S1_padded, S2_padded

def compare_spectrograms(S1, S2):
    S1_flat = S1.flatten()
    S2_flat = S2.flatten()
    similarity = np.corrcoef(S1_flat, S2_flat)[0, 1]
    return similarity

# === UI ===
st.set_page_config(page_title="EyeSightCheck App", layout="wide")
st.title("👁 EyeSightCheck: Vision + Audio Comparison")

# Tabs
tab1, tab2, tab3 = st.tabs(["📏 Snellen Chart Test", "🎧 Audio File Comparison", "🔠 LCS Similarity Check"])

# === Tab 1: Snellen Test ===
with tab1:
    st.subheader("Snellen Chart")
    st.image("snellen_chart.jpg", width=400)

    duration = st.slider("Select recording duration (seconds)", min_value=3, max_value=25, value=15)

    if st.button("🎧 Start Recording"):
        st.info(f"Recording for {duration} seconds...")

        audio = pyaudio.PyAudio()
        stream = audio.open(format=FORMAT,
                            channels=CHANNELS,
                            rate=RATE,
                            input=True,
                            frames_per_buffer=CHUNK)

        frames = []

        for _ in range(0, int(RATE / CHUNK * duration)):
            data = stream.read(CHUNK)
            frames.append(data)

        stream.stop_stream()
        stream.close()
        audio.terminate()

        with wave.open(OUTPUT_FILENAME, 'wb') as waveFile:
            waveFile.setnchannels(CHANNELS)
            waveFile.setsampwidth(audio.get_sample_size(FORMAT))
            waveFile.setframerate(RATE)
            waveFile.writeframes(b''.join(frames))

        st.success(f"✅ Recording saved! Duration: {duration} seconds")

        with st.spinner("🔎 Transcribing..."):
            model = whisper.load_model("turbo")
            result = model.transcribe(OUTPUT_FILENAME)
            transcription = result["text"].strip().upper()
            
            st.session_state["transcription"] = transcription

        st.subheader("📝 Transcription:")
        st.code(transcription)

        st.subheader("📊 Accuracy Analysis:")
        
        final_accuracy = analyze_transcription(transcription)
        st.success(f"✅ *Overall Accuracy: {final_accuracy:.2f}%*")

# === Tab 2: Audio Comparison ===
with tab2:
    st.subheader("Snellen Chart")
    st.image("snellen_chart.jpg", width=400)
    st.subheader("🎧 Upload & Compare Two Audio Files")

    audio_file1 = st.file_uploader("Upload Audio File 1", type=["mp3", "wav"], key="audio1")
    audio_file2 = st.file_uploader("Upload Audio File 2", type=["mp3", "wav"], key="audio2")

    if audio_file1 and audio_file2:
        y1, sr1 = load_audio(audio_file1)
        y2, sr2 = load_audio(audio_file2)

        S1 = compute_spectrogram(y1, sr1)
        S2 = compute_spectrogram(y2, sr2)

        S1_padded, S2_padded = pad_spectrogram(S1, S2)

        st.subheader("Spectrograms")
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        librosa.display.specshow(S1_padded, sr=sr1, x_axis='time', y_axis='log', ax=axes[0])
        axes[0].set_title("Spectrogram of Audio 1")

        librosa.display.specshow(S2_padded, sr=sr2, x_axis='time', y_axis='log', ax=axes[1])
        axes[1].set_title("Spectrogram of Audio 2")

        st.pyplot(fig)

        similarity = compare_spectrograms(S1_padded, S2_padded)
        st.subheader("Comparison Results")
        st.success(f"🎯 Spectrogram Similarity: {similarity * 100:.2f}%")

# === Tab 3: LCS Comparison ===
with tab3:
    st.subheader("🔠 LCS Similarity Analysis Between Transcription & Snellen Chart")

    original_text = "EFPTOZLPEDPECFDEDFCZPFELOPZDDEFPOTEC"

    if "transcription" not in st.session_state:
        st.warning("⚠ Please complete the Snellen Chart test in Tab 1 first.")
    else:
        transcription = transcriptionlcs

        matcher = difflib.SequenceMatcher(None, original_text, transcription)
        lcs_size = sum(block.size for block in matcher.get_matching_blocks())
        lcs_ratio = lcs_size / len(original_text) * 100 if original_text else 0

        st.markdown(f"📋 *Original Snellen Text:* {original_text}")
        st.markdown(f"📝 *Transcribed Text:* {transcription}")

        st.subheader("📊 LCS Match Result:")
        st.success(f"🔗 *Longest Common Subsequence Match: {lcs_ratio:.2f}%*")

        if(abs(lcs_ratio-final)>20):
            st.warning(f"⚠You Might Have Missed Some Letters in between we recommend a Re-test!")