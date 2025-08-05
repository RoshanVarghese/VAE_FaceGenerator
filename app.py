# app.py
import gradio as gr
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# Load the trained VAE model
vae = load_model("vae_face_model.keras", compile=False)

def reconstruct_face(input_image):
    # Resize and normalize image
    image = input_image.resize((64, 64))
    image_array = np.array(image).astype("float32") / 255.0
    image_array = np.expand_dims(image_array, axis=0)  # Shape: (1, 64, 64, 3)

    # Reconstruct using the VAE
    reconstructed = vae.predict(image_array)[0]  # Output shape: (64, 64, 3)
    reconstructed = (reconstructed * 255).astype(np.uint8)

    return Image.fromarray(reconstructed)

# Create Gradio interface
iface = gr.Interface(
    fn=reconstruct_face,
    inputs=gr.Image(type="pil"),
    outputs=gr.Image(type="pil"),
    title="VAE Face Reconstructor",
    description="Upload a face image to see its reconstruction using a Variational Autoencoder (VAE)."
)

if __name__ == "__main__":
    iface.launch()
