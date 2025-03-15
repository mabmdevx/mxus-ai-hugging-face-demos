from helper import load_image_from_url, render_results_in_image
from transformers import pipeline

from transformers.utils import logging
logging.set_verbosity_error()

from helper import ignore_warnings
ignore_warnings()

od_pipe = pipeline("object-detection", "facebook/detr-resnet-50")

from PIL import Image
raw_image = Image.open('input/sample_image1.jpeg')
raw_image.resize((569, 491))

pipeline_output = od_pipe(raw_image)

processed_image = render_results_in_image(
    raw_image, 
    pipeline_output)

# Save the processed image
processed_image = processed_image.convert('RGB')  # Convert to RGB mode
processed_image.save('output/detected_image.jpeg')