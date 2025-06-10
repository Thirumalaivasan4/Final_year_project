import streamlit as st
from PIL import Image
import google.generativeai as genai
import cv2
import numpy as np
from ultralytics import YOLO


genai.configure(api_key="AIzaSyDcUKSyQGOwbpAsaWuSOMNHJSMTKPzd0ro")  # Replace with your Gemini API key
model_gemini = genai.GenerativeModel("gemini-1.5-flash")
yolo_model = YOLO("best.pt")  # Use a small YOLOv8 model for fast inference

st.set_page_config(layout="wide")
st.title("🔧 AI-Powered Damage Detection and Repair Suggestion")


uploaded_file = st.file_uploader("📤 Upload a damaged object image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image_pil = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image_pil)

  
    results = yolo_model.predict(image_np, conf=0.3)
    annotated_frame = results[0].plot()

  
    col1, col2 = st.columns([1.3, 2])

    with col1:
        st.markdown("### 📷 YOLO Detection")
        st.image(annotated_frame, caption="Detected Image with Bounding Boxes", use_column_width=True)

    with col2:
        st.markdown("### 🤖 AI Repair Suggestions")
        st.markdown("Click below to get recommendations based on the uploaded image.")

        if st.button("Analyze with Jarvis"):
            with st.spinner("Sending to Jarvis AI..."):
                try:
                    response = model_gemini.generate_content([
                        "Identify the visible damages in this image and suggest possible repair actions ",
                        image_pil
                    ])
                    st.success("Jarvis AI Suggestions Ready!")
                    st.markdown("#### 🔧 Suggestions:")
                    st.write(response.text)
                except Exception as e:
                    st.error(f"Error from Gemini: {e}")
                
