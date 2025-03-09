from transformers.utils import logging
logging.set_verbosity_error()

import warnings
warnings.filterwarnings("ignore", message="Using the model-agnostic default `max_length`")

from transformers import BlipForConditionalGeneration
from transformers import AutoProcessor

model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")

from PIL import Image
image = Image.open("input/sample_image1.jpeg")

# Unconditional Image Captioning
inputs = processor(image,return_tensors="pt")
out = model.generate(**inputs)
print(processor.decode(out[0], skip_special_tokens=True))


# Conditional Image Captioning
text = "a photograph of"
inputs = processor(image, text, return_tensors="pt")
out = model.generate(**inputs)
print(processor.decode(out[0], skip_special_tokens=True))