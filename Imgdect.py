import streamlit as st
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
import wikipedia
import re

# Initialize BLIP model + processor (this may take a moment)
@st.cache_resource(show_spinner=False)
def load_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

processor, model = load_model()

st.title("🖼️ Image Content Detector & Info Explorer")

uploaded_file = st.file_uploader("Upload an image (PNG, JPG, JPEG)", type=["png", "jpg", "jpeg"])

def extract_keywords(text, max_keywords=5):
    # Simple keyword extractor: remove punctuation, lowercase, split words, remove stopwords
    stopwords = set([
        'the', 'a', 'an', 'in', 'on', 'with', 'and', 'is', 'are', 'of', 'to', 'at',
        'for', 'by', 'from', 'that', 'this', 'it', 'as', 'was', 'were', 'be'
    ])
    text = re.sub(r'[^\w\s]', '', text.lower())
    words = text.split()
    keywords = [w for w in words if w not in stopwords]
    # Return most frequent keywords (naive: first N unique words)
    seen = set()
    unique_keywords = []
    for w in keywords:
        if w not in seen:
            unique_keywords.append(w)
            seen.add(w)
        if len(unique_keywords) >= max_keywords:
            break
    return unique_keywords

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Analyzing image..."):
        inputs = processor(image, return_tensors="pt")
        out = model.generate(**inputs)
        caption = processor.decode(out[0], skip_special_tokens=True)

    st.markdown("### Detected Description:")
    st.write(f"**{caption.capitalize()}**")

    keywords = extract_keywords(caption)
    if keywords:
        st.markdown("### Detailed Information from Wikipedia:")
        for word in keywords:
            try:
                summary = wikipedia.summary(word, sentences=2)
                st.markdown(f"**{word.capitalize()}:** {summary}")
            except Exception as e:
                st.markdown(f"**{word.capitalize()}:** No detailed info found.")
    else:
        st.write("No keywords found to fetch detailed information.")
