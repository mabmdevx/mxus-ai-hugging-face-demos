import streamlit as st
from transformers import BartForConditionalGeneration, BartTokenizer
import torch

# Load the model and tokenizer
model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")

def summarize_text(text):
    # Tokenize the text
    inputs = tokenizer(text, return_tensors="pt")

    # Generate summary
    summary_ids = model.generate(inputs["input_ids"], 
                                    num_beams=4,
                                    min_length=20, 
                                    early_stopping=True)

    # Convert summary IDs to text
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    return summary

st.title("Text Summarizer")
st.write("Enter text to generate a summary")

text_input = st.text_area(label="Text to summarize", height=200)
if st.button("Summarize"):
    summary = summarize_text(text_input)
    st.info(f"**Summary:**\n{summary}")