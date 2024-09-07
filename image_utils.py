import os
import dotenv

dotenv.load_dotenv()

from cloudflare import Cloudflare
from IPython.display import display, Image
import streamlit as st

ACCOUNT_ID = st.secrets["ACCOUNT_ID"]
CLOUDFLARE_API_TOKEN = st.secrets["CLOUDFLARE_API_TOKEN"]
MODEL_NAME = st.secrets["MODEL_NAME"]

client = Cloudflare(api_token=st.secrets["CLOUDFLARE_API_TOKEN"])

def generate_and_display_image_from_summary(summary: str): 
    """
    Generate and display an image with @cf/bytedance/stable-diffusion-xl-lightning via Cloudflare Workers AI API.
    """
    data = client.workers.ai.with_raw_response.run(
        model_name=MODEL_NAME,
        account_id=ACCOUNT_ID,
        prompt=summary
    )
    display(Image(data.read()))


def generate_and_save_image_from_summary(summary: str, file_path: str):
    """
    Generates and saves an image with @cf/bytedance/stable-diffusion-xl-lightning via Cloudflare Workers AI API.
    """
    data = client.workers.ai.with_raw_response.run(
        model_name=MODEL_NAME,
        account_id=ACCOUNT_ID,
        prompt=summary
    )
    with open(f"{file_path}/stable_diffusion.png", "wb") as outfile: 
        outfile.write(data.read())
