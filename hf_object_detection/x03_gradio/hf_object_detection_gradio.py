import gradio as gr
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

# Define the Gradio interface
demo = gr.Interface(
    fn=detect_objects,
    inputs=gr.Image(type="pil"),
    outputs=gr.Image(type="pil"),
    title="Object Detection",
    description="Detect objects in an image",
)

# Launch the Gradio app
demo.launch()