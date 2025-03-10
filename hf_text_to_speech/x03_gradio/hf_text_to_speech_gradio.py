import gradio as gr
import torch
import numpy as np
import scipy
from transformers import VitsModel, AutoTokenizer

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

demo = gr.Interface(
    text_to_speech,
    gr.Textbox(label="Text to narrate"),
    gr.Audio(label="Narrated audio"),
    title="Text-to-Speech",
    description="Enter text to generate audio narration",
)

demo.launch()