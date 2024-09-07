import os
import dotenv

dotenv.load_dotenv()

from cloudflare import Cloudflare
# from IPython.display import display, Image
import streamlit as st
import requests
from PIL import Image
from io import BytesIO

ACCOUNT_ID = st.secrets["ACCOUNT_ID"]
CLOUDFLARE_API_TOKEN = st.secrets["CLOUDFLARE_API_TOKEN"]
MODEL_NAME = st.secrets["MODEL_NAME"]

client = Cloudflare(api_token=st.secrets["CLOUDFLARE_API_TOKEN"])

def generate_and_display_image_from_summary(summary: str, save: bool = False, file_path: str = 'images'):
    """
    Generate and display an image with @cf/bytedance/stable-diffusion-xl-lightning via Cloudflare Workers AI API.
    Optionally save the image to a file.
    """
    try:
        # Step 1: Generate the image with the API
        data = client.workers.ai.with_raw_response.run(
            model_name=MODEL_NAME,
            account_id=ACCOUNT_ID,
            prompt=summary
        )
        
        # Convert the response to an image
        image = Image.open(BytesIO(data.read()))

        image = image.resize((700, 700))
        
        # Step 2: Display the image using Streamlit
        st.image(image, caption="Generated Image", use_column_width=False)
    
    except Exception as e:
        st.error(f"Error generating or displaying image: {e}")


