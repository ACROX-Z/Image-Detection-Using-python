import streamlit as st
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from google.cloud import aiplatform
from google.oauth2 import service_account

# --- CONFIGURATION: Set your own values here ---
PROJECT_ID = "your-gcp-project-id"
LOCATION = "us-central1"
ENDPOINT_ID = "your-gemini-endpoint-id"  # e.g. "1234567890123456789"
SERVICE_ACCOUNT_FILE = "path/to/your/service-account.json"

# Initialize credentials and AI Platform client
credentials = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE)
client = aiplatform.gapic.PredictionServiceClient(credentials=credentials)
endpoint = client.endpoint_path(project=PROJECT_ID, location=LOCATION, endpoint=ENDPOINT_ID)

@st.cache_resource(show_spinner=False)
def load_blip_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

processor, model = load_blip_model()

st.title("Image Caption + Detailed Info via Gemini API")

uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

def generate_caption(image):
    inputs = processor(image, return_tensors="pt")
    out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption

def ask_gemini(prompt_text):
    instances = [{"content": prompt_text}]
    parameters = {"temperature": 0.7, "maxOutputTokens": 500}

    response = client.predict(
        endpoint=endpoint,
        instances=instances,
        parameters=parameters,
    )
    # The response's predictions contain the generated text
    prediction = response.predictions[0].get("content", "")
    return prediction

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Generating image caption..."):
        caption = generate_caption(image)

    st.markdown("### Image Caption:")
    st.write(f"**{caption.capitalize()}**")

    if st.button("Get More Information from Gemini"):
        with st.spinner("Contacting Gemini API..."):
            prompt = (
                "You are a helpful assistant that provides detailed information about an image. "
                "Here is the image description: "
                f"'{caption}'. Please give a detailed explanation and any interesting facts about it."
            )
            detailed_response = ask_gemini(prompt)

        st.markdown("### Gemini's Detailed Explanation:")
        st.write(detailed_response)
