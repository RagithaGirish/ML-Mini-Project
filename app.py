import os
import tempfile
import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms

# Page configuration
st.set_page_config(
    page_title="Cervical Cell Classifier",
    page_icon="🔬",
    layout="centered"
)

# Load checkpoint
@st.cache_resource
def load_model():
    checkpoint = torch.load("results/baseline_resnet/best_model.pt", map_location="cpu")
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, len(checkpoint["class_names"]))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint["class_names"], checkpoint["image_size"]

def predict_image(model, image_path, class_names, image_size):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))
    ])
    image = Image.open(image_path).convert("RGB")
    img_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)[0]
        pred_idx = torch.argmax(probs).item()
    return class_names[pred_idx], probs.numpy()

def main():
    st.title("Cervical Cell Image Classification Using ResNet-18")

    # st.warning("""
    # **IMPORTANT DISCLAIMER:**
    # This application is developed for educational and research purposes only. 
    # It is not a medical diagnostic device and should not be used for clinical decision-making.
    # """)

    st.sidebar.header("About the Model")
    st.sidebar.info("""
    **Model Type:** Pretrained ResNet-18  
    **Training:** Fine-tuned on SIPaKMeD cropped dataset  
    **Number of Classes:** 5  
    - Dyskeratotic
    - Koilocytotic
    - Metaplastic
    - Parabasal
    - Superficial-Intermediate
    """)

    try:
        model, class_names, image_size = load_model()
    except Exception as e:
        st.error(f"Error loading ResNet model. Ensure best_model.pt exists in results/baseline_resnet. Details: {e}")
        return

    st.subheader("1. Upload a Cell Image")
    uploaded_file = st.file_uploader("Supported Formats: BMP, PNG, JPG, JPEG", type=["bmp", "png", "jpg", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_container_width=True)

        st.write("---")
        st.subheader("2. Run Analysis")

        if st.button("Run Prediction", type="primary"):
            with st.spinner("Running ResNet prediction..."):
                file_extension = os.path.splitext(uploaded_file.name)[1]
                with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = tmp_file.name

                try:
                    predicted_class, probabilities = predict_image(model, tmp_path, class_names, image_size)
                    st.success(f"### Predicted Class: **{predicted_class}**")

                    st.write("#### Class Probabilities")
                    prob_dict = {class_names[i]: float(probabilities[i]) for i in range(len(class_names))}
                    st.bar_chart(prob_dict)

                finally:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)

if __name__ == "__main__":
    main()
