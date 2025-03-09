import streamlit as st
import torch
import numpy as np
import scipy
from transformers import VitsModel, AutoTokenizer

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

# Load the model and tokenizer
model = VitsModel.from_pretrained("kakao-enterprise/vits-ljs")
tokenizer = AutoTokenizer.from_pretrained("kakao-enterprise/vits-ljs")

def text_to_speech(text):
    # Tokenize the text
    inputs = tokenizer(text, return_tensors="pt")

    # Generate audio
    with torch.no_grad():
        output = model(**inputs).waveform

    # Convert to numpy array and save as WAV file
    audio_array = output.cpu().numpy().squeeze()
    audio_array /= 1.414
    audio_array *= 32767
    audio_array = audio_array.astype(np.int16)

    # Save to WAV file
    output_file = "output/output.wav"
    scipy.io.wavfile.write(output_file, rate=model.config.sampling_rate, data=audio_array)

    # Return the path to the WAV file
    return output_file

if __name__ == "__main__":
    st.title("Text-to-Speech Demo")
    st.write("Enter text to generate audio narration")

    text_input = st.text_area(label="Text to narrate")

    if st.button("Generate Audio"):
        output_file = text_to_speech(text_input)
        audio_file = open(output_file, "rb")
        audio_bytes = audio_file.read()
        st.audio(audio_bytes, format="audio/wav")