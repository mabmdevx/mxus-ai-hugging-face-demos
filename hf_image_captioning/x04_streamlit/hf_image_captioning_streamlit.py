import streamlit as st
from transformers import BlipForConditionalGeneration, AutoProcessor
from PIL import Image
import torch

# Load model and processor
@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_model():
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    return model, processor

model, processor = load_model()

def generate_caption(image, text=""):
    # Preprocess image
    inputs = processor(images=image, text=text, return_tensors="pt")
    
    # Generate caption
    with torch.no_grad():
        out = model.generate(**inputs)
    
    # Postprocess caption
    caption = processor.decode(out[0], skip_special_tokens=True)
    
    return caption

# Create Streamlit interface
st.title("Image Captioning with BLIP")
st.write("Generate captions for images using the BLIP model.")

# Upload image
uploaded_image = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_image:
    # Display uploaded image
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image")

    # Provide optional context
    context = st.text_input("Optional Context (e.g., 'a photograph of')")

    # Generate caption
    if st.button("Generate Caption"):
        caption = generate_caption(image, context)
        st.write("Generated Caption:")
        st.info(caption)