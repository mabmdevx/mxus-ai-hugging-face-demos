import streamlit as st
from transformers import pipeline
from PIL import Image
from helper import load_image_from_url, render_results_in_image

# Load the object detection pipeline
od_pipe = pipeline("object-detection", "facebook/detr-resnet-50")

def detect_objects(image):
    # Run object detection
    pipeline_output = od_pipe(image)

    # Render results in the image
    processed_image = render_results_in_image(image, pipeline_output)

    return processed_image

# Streamlit app
st.title("Object Detection")
st.write("Detect objects in an image")

# File uploader for image input
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load the image
    image = Image.open(uploaded_file)

    # Detect objects in the image
    processed_image = detect_objects(image)

    # Display the original and processed images
    st.image([image, processed_image], caption=["Original Image", "Processed Image"])
