import gradio as gr
from transformers import BlipForConditionalGeneration, AutoProcessor
from PIL import Image

# Load model and processor
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")

def generate_caption(image, text=""):
    # Preprocess image
    inputs = processor(images=image, text=text, return_tensors="pt")
    
    # Generate caption
    out = model.generate(**inputs)
    
    # Postprocess caption
    caption = processor.decode(out[0], skip_special_tokens=True)
    
    return caption

# Create Gradio interface
demo = gr.Interface(
    fn=generate_caption,
    inputs=[
        gr.Image(type="pil", label="Image"),
        gr.Textbox(label="Optional Context (e.g., 'a photograph of')"),
    ],
    outputs=gr.Textbox(label="Generated Caption"),
    title="Image Captioning",
    description="Generate captions for images using the BLIP model.",
)

# Launch Gradio interface
demo.launch()