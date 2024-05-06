from flask import Flask, request, send_file
import torch
from diffusers import StableDiffusionPipeline
from io import BytesIO

app = Flask(__name__)

# Load the model
pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16)
pipe = pipe.to("cuda")

@app.route('/generate_image', methods=['POST'])
def generate_image():
    # Get the prompt from the request
    prompt = request.json.get('prompt', '')
    
    # Generate the image
    image = pipe(prompt).images[0]
    
    # Convert the image to PNG format
    image_bytes = BytesIO()
    image.save(image_bytes, format='PNG')
    image_bytes.seek(0)
    
    # Return the image
    return send_file(image_bytes, mimetype='image/png')


if __name__ == "__main__":
    # Run the Application
    app.run(debug=True)
