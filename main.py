import streamlit as st
from app import load_app
from integration import load_integration
from PIL import Image

st.set_page_config(page_title="Golden Gate Bridge Themed App", layout="wide")

# Load image with PIL
image = Image.open("spooks.png")

# Resize image to specific height while maintaining aspect ratio
resized_image = image.resize((1280, 300))

# Display the resized image
st.image(resized_image)
st.title("👁️‍🗨️🌉 SF Conspiracy Theory Generator ")

# Creating tabs
tab1, tab2, tab3 = st.tabs(["Generate Conspiracies", "Mic", 'About the Project'])

# Generate Conspiracies tab content
with tab1:
    st.header("Generate Conspiracies")
    st.write( "This is a chatbot powered by OpenAI's GPT-3.5-Turbo, orchestrated by Haystack 2.0 to generate conspiracy theories about the city of San Francisco.")
    load_app()
# Mic content
with tab2:
    st.header("Mic Overview")
    st.write("This tab can be used to display mic version or images maybe.")
    # Define a function to handle recording logic
    load_integration()

# About the Project content
with tab3:
    st.header("About This App")
    st.write("This tab provides information about the app.")
    st.write("You can use this section to explain the purpose of the app, its features, or any other relevant information.")
    # app.py
    text = st.text_area("Enter text to format into a newspaper:", "This is a sample text to format into a newspaper style layout. Feel free to adjust this text.")


