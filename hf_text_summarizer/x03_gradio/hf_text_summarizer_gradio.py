import gradio as gr
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
                                    #min_length=20, 
                                    #min_length=50, 
                                    early_stopping=True)

    # Convert summary IDs to text
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    return summary

demo = gr.Interface(
    fn=summarize_text,
    inputs=gr.Textbox(label="Text to summarize"),
    outputs=gr.Textbox(label="Summary"),
    title="Text Summarizer",
    description="Enter text to generate a summary",
)

demo.launch()