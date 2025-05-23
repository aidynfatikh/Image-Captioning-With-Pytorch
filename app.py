import os
import streamlit as st
import pickle

import torchvision.transforms as transforms
from PIL import Image
import torch
import model

from config import *

with open("dataset/vocab.pkl", "rb") as f:
    vocab = pickle.load(f)

vocab_size = len(vocab.stoi)
transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

def load_checkpoint(model, optimizer=None, path=CHECKPOINT_PATH):
    ckpt = torch.load(path, weights_only=False)
    model.load_state_dict(ckpt['model'])
    if optimizer:
        optimizer.load_state_dict(ckpt['optimizer'])
    return ckpt['step']

def closest_power(n):
    if n < 1:
        raise ValueError("input must be a positive integer.")
    return 1 << (n - 1).bit_length()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"current device: {device.upper()}")
modelconfig = model.ITTConfig()
modelconfig.vocab_size = closest_power(vocab_size)
ITT = model.ITT(modelconfig).to(device)
ITT = torch.compile(ITT)
torch.set_float32_matmul_precision("high")
if os.path.exists(CHECKPOINT_PATH):
    start_step = load_checkpoint(ITT, path=CHECKPOINT_PATH)
    print(f"loaded checkpoint at step {start_step}\n")

ITT.eval()
st.title("Image Captioner")
st.write("Upload an image to generate a caption.")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # ensure session state is initialized
    if "last_uploaded_file" not in st.session_state:
        st.session_state.last_uploaded_file = None
        st.session_state.caption = ""

    # check if a new image is uploaded
    if uploaded_file.name != st.session_state.last_uploaded_file:
        st.session_state.last_uploaded_file = uploaded_file.name  
        st.session_state.caption = ""  # reset caption

    # generate caption if its empty
    if not st.session_state.caption:
        with torch.no_grad():
            image_tensor = transform(image).unsqueeze(0).to(device)

            output = ITT.sample(image_tensor, vocab, 20)
            caption = " ".join(output)

        st.session_state.caption = caption  

    st.subheader("Generated Caption:")
    st.write(st.session_state.caption)
