# VAE Face Generator

This project uses a Variational Autoencoder (VAE) to reconstruct and generate face images.

## 🔍 What It Does

- Reconstructs uploaded face images
- Generates new random faces from latent space
- Interface built with Gradio

---

## Run in Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RoshanVarghese/VAE_FaceGenerator/blob/main/GenAI_VAE_GitHub.ipynb)

---

## Files

- `app.py`: Gradio interface
- `vae_model.py`: Custom VAE class
- `vae_face_model.keras`: Pretrained VAE model
- `run_vae.ipynb`: Launches the model from Colab
- `requirements.txt`: Dependencies

---

## Local Setup

```bash
git clone https://github.com/your-username/vae-face-generator.git
cd vae-face-generator
pip install -r requirements.txt
python app.py
