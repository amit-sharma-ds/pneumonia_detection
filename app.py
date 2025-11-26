import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image

# -----------------------------
# Load Model From Local File
# -----------------------------
@st.cache_resource
def load_pneumonia_model():
    model = load_model("model.h5")
    return model

model = load_pneumonia_model()

# -----------------------------
# Preprocessing Function
# -----------------------------
def preprocess_image(image):
    # Resize to 224x224 (CNN input)
    image = image.resize((224, 224))  
    # Convert to grayscale
    image = image.convert("L")        
    # Convert to numpy array
    image = np.array(image)
    # Normalize
    image = image / 255.0             
    # Expand dimensions
    image = np.expand_dims(image, axis=-1)
    # Convert 1-channel → 3-channel
    image = np.repeat(image, 3, axis=-1)
    # Batch dimension
    image = np.expand_dims(image, axis=0)
    return image

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🩺 Pneumonia Detection using CNN")
st.write("Upload a Chest X-ray image to check whether pneumonia is present or not.")

uploaded_file = st.file_uploader("Upload X-ray Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded X-ray", use_column_width=True)

    # Preprocess the image
    img = preprocess_image(image)

    # Make prediction
    pred = model.predict(img)[0][0]
    result = "PNEUMONIA DETECTED" if pred > 0.5 else "NORMAL"

    st.subheader("🔍 Prediction Result:")
    st.write(f"**Model Output Score:** {pred:.4f}")

    # -----------------------------
    # RESULT DISPLAY TEXT
    # -----------------------------
    if result == "PNEUMONIA DETECTED":
        st.error("⚠️ **PNEUMONIA DETECTED**")
        st.markdown(
            """
- The model suggests the presence of pneumonia in the chest X-ray.
- Please consult a medical professional for accurate diagnosis.
- Early detection can help in better treatment and management.
            """
        )
    else:
        st.success("✅ **NORMAL – No Pneumonia Detected**")
        st.markdown(
            """
- The model indicates that the X-ray appears normal.
- No signs of pneumonia were detected.
- If symptoms persist, a medical checkup is still recommended.
            """
        )
