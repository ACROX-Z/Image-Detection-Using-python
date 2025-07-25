# Image-Detection-Using-python
# This is a simple image detection using python and OpenCV library
# Image Content Detection & Detailed Information Explorer

This repository contains two Streamlit-based Python applications that perform image content detection and provide detailed information about the detected content.

---

## 1. Image Content Detector with Wikipedia Info

### Overview

This app uses the BLIP model to generate a caption for an uploaded image. It then extracts keywords from the caption and fetches relevant summaries from Wikipedia for those keywords, displaying detailed information alongside the image.

### Features

- Upload an image (PNG, JPG, JPEG).
- Automatically generate an image caption using the BLIP model.
- Extract important keywords from the caption.
- Fetch and display short Wikipedia summaries for each keyword.

### How to Run

1. Install dependencies:

   \\bash
   pip install streamlit transformers torch pillow wikipedia

Run the app:

    \\bash
    streamlit run image_detector_wiki.py

 ### Dependencies
    1.streamlit
    2.transformers
    3.torch
    4.Pillow
    5.wikipedia

### 2. Image Caption + Detailed Information via Google Gemini API
Overview
This app also uses the BLIP model to generate captions for uploaded images. However, it sends the caption as a prompt to Google’s Gemini API via Vertex AI to get a more detailed and contextual explanation.

Features:
    Upload an image (PNG, JPG, JPEG).
    Generate an image caption with BLIP.
    Query Google Gemini API for detailed image-related information.
    Display Gemini's detailed explanation.

Setup Requirements :
    Google Cloud project with billing enabled.
    Vertex AI API enabled.
    Gemini (PaLM 2) model endpoint created in Vertex AI.
    Service account JSON key with appropriate permissions.

How to Run
Install dependencies:

    \\bash
        pip install streamlit transformers torch pillow google-cloud-aiplatform google-auth
Set your Google Cloud project details and service account JSON path in the script:

### python
PROJECT_ID = "your-gcp-project-id"
LOCATION = "us-central1"
ENDPOINT_ID = "your-gemini-endpoint-id"
SERVICE_ACCOUNT_FILE = "path/to/your/service-account.json"
Run the app:

    \\bash
        streamlit run image_caption_gemini.py