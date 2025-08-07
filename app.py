import gradio as gr
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from vae_model import VAE  # Your custom model class

# Load the full model
vae = load_model("vae_face_model.keras", compile=False, custom_objects={"VAE": VAE})

# Extract decoder
decoder = vae.decoder  # ✅ This assumes your VAE object has a .decoder attribute

# Function to generate a random face
def generate_face():
    # Sample a random latent vector from standard normal distribution
    latent_dim = decoder.input_shape[1]  # Example: (None, 128) -> latent_dim = 128
    z = np.random.normal(size=(1, latent_dim)).astype("float32")

    # Generate image
    generated = decoder.predict(z)[0]  # Shape: (64, 64, 3)
    generated = (generated * 255).astype(np.uint8)
    return Image.fromarray(generated)

# Function to reconstruct an uploaded face
def reconstruct_face(input_image):
    image = input_image.resize((64, 64))
    image_array = np.array(image).astype("float32") / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    reconstructed = vae.predict(image_array)[0]
    reconstructed = (reconstructed * 255).astype(np.uint8)
    return Image.fromarray(reconstructed)

# Gradio Interface
reconstruct_interface = gr.Interface(
    fn=reconstruct_face,
    inputs=gr.Image(type="pil"),
    outputs=gr.Image(type="pil"),
    title="Reconstruct Face",
    description="Upload a face to reconstruct it using the VAE."
)

generate_interface = gr.Interface(
    fn=generate_face,
    inputs=[],
    outputs=gr.Image(type="pil"),
    title="Generate Face",
    description="Click to generate a new face using the decoder from the VAE."
)

# Combine both tabs
iface = gr.TabbedInterface(
    interface_list=[reconstruct_interface, generate_interface],
    tab_names=["Reconstruct Face", "Generate New Face"]
)

if __name__ == "__main__":
    iface.launch(share=True)
