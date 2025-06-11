# app.py
import streamlit as st
from PIL import Image
import torch
from model import get_model, preprocess_image

st.set_page_config(page_title="DR Classifier", layout="centered")
st.title("🩺 Diabetic Retinopathy Classifier (Ensemble)")

@st.cache_resource
def load_models():
    resnet = get_model("resnet34", "resnet34_best.zip", is_zipped=True)
    effnet = get_model("efficientnet_b0", "efficientnet_b0_best.pth")
    return [resnet, effnet]

models = load_models()

def ensemble_predict(models, x):
    with torch.no_grad():
        outputs = [m(x) for m in models]
        avg_logits = torch.stack(outputs).mean(0)
        return torch.argmax(avg_logits, dim=1).item()

uploaded = st.file_uploader("Upload a retina image", type=["jpg", "jpeg", "png"])
if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded Retina Image", use_column_width=True)
    input_tensor = preprocess_image(image)
    prediction = ensemble_predict(models, input_tensor)
    class_names = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]
    st.success(f"Prediction: **{class_names[prediction]}** (class {prediction})")

