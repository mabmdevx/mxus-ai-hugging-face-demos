from transformers import VitsModel, AutoTokenizer
import torch
import numpy as np
import scipy

model = VitsModel.from_pretrained("kakao-enterprise/vits-ljs")
tokenizer = AutoTokenizer.from_pretrained("kakao-enterprise/vits-ljs")

text = "Hey, it's Hugging Face Text to Speech running locally"
inputs = tokenizer(text, return_tensors="pt")

with torch.no_grad():
    output = model(**inputs).waveform


audio_array = output.cpu().numpy().squeeze()
audio_array /=1.414
audio_array *= 32767
audio_array = audio_array.astype(np.int16)
# print(audio_array)

scipy.io.wavfile.write("output/output.wav", rate=model.config.sampling_rate, data=audio_array)
