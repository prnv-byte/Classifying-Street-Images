import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# --- 1. Load the Professional Google Brain ---
@st.cache_resource 
def load_professional_model():
    # We are downloading MobileNetV2, pre-trained on 1.4 million images (ImageNet)
    model = tf.keras.applications.MobileNetV2(weights='imagenet')
    return model

# --- 2. Image Processing Pipeline ---
def process_image(img):
    # 1. Convert to RGB
    img = img.convert('RGB')
    
    # 2. MobileNetV2 expects MUCH higher resolution images: 224x224 pixels!
    img = img.resize((224, 224))
    
    # 3. Convert to numpy array
    img_array = np.array(img)
    
    # 4. Create the batch dimension (1, 224, 224, 3)
    img_array = np.expand_dims(img_array, axis=0)
    
    # 5. MobileNet has a special mathematical preprocessing function we MUST use
    processed_image = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    
    return processed_image

# --- 3. Streamlit Web Interface ---
st.title("🌍 World-Class Vision AI")
st.write("This AI is powered by MobileNetV2 and can recognize **1,000 different objects** in high resolution!")

# Load the massive model (This might take a few seconds the very first time!)
model = load_professional_model()

# File uploader widget
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_container_width=True)
    
    st.write("Consulting the 1.4-million-image brain...")
    
    # Process the image
    processed_image = process_image(image)
    
    # Get predictions (An array of 1,000 probabilities!)
    predictions = model.predict(processed_image)
    
    # TensorFlow has a built-in translator to turn those 1,000 numbers into English words
    decoded_predictions = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=5)[0]
    
    # Get the absolute best guess
    best_id, best_label, best_confidence = decoded_predictions[0]
    
    # Display Results
    st.divider()
    
    # Replace underscores with spaces to make it look nicer (e.g., 'street_sign' -> 'street sign')
    formatted_label = best_label.replace('_', ' ').title()
    
    st.success(f"🤖 **Top Prediction:** {formatted_label}")
    st.info(f"**Confidence:** {best_confidence:.2%}")
    
    # Show the top 5 runner-up guesses
    with st.expander("See the AI's top 5 guesses"):
        for pred_id, label, prob in decoded_predictions:
            st.write(f"- **{label.replace('_', ' ').title()}**: {prob:.2%}")