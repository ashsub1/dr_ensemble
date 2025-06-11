import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
from model import get_model

st.set_page_config(page_title="DR Ensemble Classifier", layout="centered")

# ✅ Preprocessing
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# ✅ Load models (cached)
@st.cache_resource
def load_models():
    resnet = get_model("resnet34", "resnet34_best.zip", is_zipped=True)
    efficientnet = get_model("efficientnet_b0", "efficientnet_b0_best.pth")
    return [resnet, efficientnet]

models = load_models()

# ✅ Inference
def ensemble_predict(models, x):
    logits = [m(x) for m in models]
    return torch.stack(logits).mean(0)

# ✅ UI
st.title("📈 Diabetic Retinopathy Classification (Ensemble)")

uploaded_file = st.file_uploader("Upload a Retina Image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img_tensor = transform(image).unsqueeze(0)  # [1, 3, 224, 224]

    with st.spinner("Classifying..."):
        with torch.no_grad():
            preds = ensemble_predict(models, img_tensor)
            predicted_class = preds.argmax(1).item()
    
    class_names = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]
    st.success(f"🔍 Predicted: **{class_names[predicted_class]}**")
