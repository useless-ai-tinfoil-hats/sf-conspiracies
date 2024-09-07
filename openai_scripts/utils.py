from io import BytesIO
from typing import Optional, Union
import requests


from PIL import Image
import matplotlib.pyplot as plt


from openai import OpenAI


def create_intro_paragraph(client: OpenAI, user_prompt: str = "") -> str: 
    """takes a user prompt about a conspiracy theory and returns text about that theory"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": """
                You are a seasoned conspiracy theorest, skilled at off-the-cuff improvisation when creating new theories. 
                You speak with the tone and clarity of a news reporter, and incorporate details a user gives you to create news 
                articles about conspiracies in the San Francisco Bay Area. You always tie your conspiracies to the SF area.
                This response should be techy but 'useless' since I'm tired of AI hype.
            """},
            {"role": "user", "content": user_prompt}
        ]
    )
    return response.choices[0].message.content


def generate_image(client: OpenAI, intro: str = "") -> str:
    """takes article text and creates image based on that content"""
    response = client.images.generate(
        model="dall-e-3",
        prompt=f"create an image inspired by this text: {intro}",
        size="1024x1024",
        quality="standard",
        n=1,
    )
    return response.data[0].url


def display_image(file_path: str = "images", image_url: str = "", save: bool = False) -> None:
    """displays and optionally saves image in image_url"""
    
    response = requests.get(image_url)
    img = Image.open(BytesIO(response.content))

    if save:
        image_name = image_url.replace("/", "+")
        img.save(f"./{file_path}/image_{image_name[-15: ]}.png")

    plt.imshow(img)
    plt.axis('off')  
    plt.show()
