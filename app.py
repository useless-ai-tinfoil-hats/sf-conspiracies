import streamlit as st
import os
import pickle
from haystack import Document
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.components.builders import PromptBuilder
from haystack.components.generators import OpenAIGenerator
from haystack import Pipeline
from dotenv import load_dotenv
import openai
import requests
from io import BytesIO
from PIL import Image

def load_app():
    load_dotenv()

    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

    if not OPENAI_API_KEY and "api_key_set" not in st.session_state:
        openai_api_key = st.text_input("Please enter your OpenAI API key:", type="password")
        if openai_api_key:
            OPENAI_API_KEY = openai_api_key
            st.session_state.api_key_set = True
            st.success("API key set successfully!")
            st.rerun()

    openai.api_key = OPENAI_API_KEY

    preprocessed_data_path = 'preprocessed_data.pkl'

    def load_preprocessed_data(path: str):
        with open(path, 'rb') as f:
            return pickle.load(f)

    document_store = InMemoryDocumentStore()

    docs_with_embeddings = load_preprocessed_data(preprocessed_data_path)
    document_store.write_documents(docs_with_embeddings["documents"])

    text_embedder = SentenceTransformersTextEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")

    retriever = InMemoryEmbeddingRetriever(document_store=document_store)

    summary_text_embedder = SentenceTransformersTextEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")


    template = """
    Using the context provided, generate a creative and outlandish conspiracy theory in a paragraph (Your total response (including title and references) MUST be 800 characters or less). 
    Your theory should be based on the following news articles about events in San Francisco from the last year.
    Make sure the theory is wild but still incorporates specific details news articles.
    At the beginning of your answer, provide a catchy title for the theory in 7 or fewer words.
    Bold this title using Markdown.
    Don't use the word "Conspiracy" in your title.
    In addition, center the title and have it on its own unique line.
    Don't have a colon at the end of the title
    At the end of your answer, provide the titles of the articles you used in a bulleted list.

    Context:
    Below are the key news articles that occurred in San Francisco over the last year. 
    Use them to inform your answer:
    {% for document in documents %}
        {{ document.content }}
    {% endfor %}

    Question: {{question}}
    Answer:
    """

    mic_template = """
    Based on the following conversation, generate a creative and outlandish conspiracy theory in a paragraph
    (Your total response (including title and references) MUST be 800 characters or less). 
    Your theory should be based on the following news articles about events in San Francisco from the last year.
    Make sure the theory is wild but still incorporates specific details news articles.
    At the beginning of your answer, provide a catchy title for the theory in 7 or fewer words.
    Bold this title using Markdown.
    Don't use the word "Conspiracy" in your title.
    In addition, center the title and have it on its own unique line.
    Don't have a colon at the end of the title
    At the end of your answer, provide the titles of the articles you used in a bulleted list.

    Context:
    Below are the key news articles that occurred in San Francisco over the last year. 
    Use them to inform your answer:
    {% for document in documents %}
        {{ document.content }}
    {% endfor %}

    Question: {{question}}
    Answer:
    """

    summary_template = """
    Given the following input with a conspiracy title, description, and sources, create a highly 
    detailed and vivid scene that can be passed into an AI image generator based on the title and description.
    The scene should include specific visual elements, actions, 
    setting, and mood that reflect the core theme of the conspiracy theory. The prompt should be 
    structured for optimal image generation.

    **Conspiracy Theory Description**:
    {{input}}

    **Requirements**:
    1. **Setting**: Where is the scene taking place? Provide a vivid description of the location, time of day, weather, or other environmental factors.
    2. **Main Characters/Objects**: In one sentence, describe what or who is in the scene? Include any important figures, objects, or entities (e.g., UFOs, secret agents, strange artifacts).
    3. **Key Actions**: What is happening? Describe any movements, interactions, or behaviors.
    4. **Atmosphere and Mood**: Set the tone of the image (e.g., eerie, mysterious, futuristic) in one sentence.

    **Example Output**:
    A futuristic night-time scene in San Francisco, with a large UFO hovering silently above the Golden Gate Bridge. The bridge is dimly lit by streetlights, casting long shadows on the road. Dark clouds swirl ominously in the sky, as the UFO emits a faint, eerie blue glow. Below, several shadowy figures dressed in black suits and sunglasses are standing near the base of the bridge, staring upwards. The city in the background is barely visible through a thick, fog-like mist. The atmosphere is tense, with cool tones dominating the scene. The lighting is dramatic, with sharp contrasts between light and dark areas.

    Now generate a similar prompt based on the given conspiracy theory.
    """


    temp_mic_prompt = """
    Speaker A: Hi, Rafa. How are you doing?
    Speaker B: Hi, Vitoria. I'm doing good. And you?
    Speaker A: I'm doing good. What have you been up to?
    Speaker B: Not much. Just walking here and there. Did you hear what happened to me the other day?
    Speaker A: No, what happened?
    Speaker B: I was walking through Union Square and I saw a ginormous quantity of pigeons.
    Speaker A: Oh my God. Yes. You know, I was at the ferry building and there were so many there too. So annoying.
    Speaker B: You know what it is? I think it's the fact that they're building now. Pigeonshe they're not really birds. They're there to monitor us. So they're cyborg agents done by the government in order to monitor the people so that they behave just like the government wants.
    Speaker A: Damn. Do you really think that would be happening?
    Speaker B: I don't know, but I don't trust the government that much, to be honest.
    Speaker A: Fair enough.
    """
    # prompt_builder = PromptBuilder(template=template)
    # basic_rag_pipeline = Pipeline()
    # basic_rag_pipeline.add_component("text_embedder", text_embedder)
    # basic_rag_pipeline.add_component("retriever", retriever)
    # basic_rag_pipeline.add_component("prompt_builder", prompt_builder)
    # basic_rag_pipeline.add_component("llm", OpenAIGenerator(model="gpt-3.5-turbo"))
    # basic_rag_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
    # basic_rag_pipeline.connect("retriever", "prompt_builder.documents")
    # basic_rag_pipeline.connect("prompt_builder", "llm")

    prompt = temp_mic_prompt
    # Function to create the prompt builder based on the selected template
    def get_prompt_builder(use_mic_template: bool):
        selected_template = mic_template if use_mic_template else template
        return PromptBuilder(template=selected_template)

    # Create pipelines dynamically based on the selected template
    def create_pipeline(use_mic_template: bool):
        prompt_builder = get_prompt_builder(use_mic_template)
        basic_rag_pipeline = Pipeline()
        basic_rag_pipeline.add_component("text_embedder", text_embedder)
        basic_rag_pipeline.add_component("retriever", retriever)
        basic_rag_pipeline.add_component("prompt_builder", prompt_builder)
        basic_rag_pipeline.add_component("llm", OpenAIGenerator(model="gpt-3.5-turbo"))
        
        # Connect pipeline components
        basic_rag_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
        basic_rag_pipeline.connect("retriever", "prompt_builder.documents")
        basic_rag_pipeline.connect("prompt_builder", "llm")
        
        return basic_rag_pipeline

    def create_summary_pipeline():
        summary_prompt_builder = PromptBuilder(template=summary_template)
        summary_rag_pipeline = Pipeline()
        summary_rag_pipeline.add_component("prompt_builder", summary_prompt_builder)
        summary_rag_pipeline.add_component("llm", OpenAIGenerator(model="gpt-3.5-turbo"))
        summary_rag_pipeline.connect("prompt_builder", "llm")
        return summary_rag_pipeline

    # 5. **Visual Details**: In a maximum of one sentence, focus on visual features such as color palette, textures, and any fine details that make the image compelling and related to the conspiracy.
    # Pipeline for creating one sentence summary
    input_pipeline = create_pipeline(True)
    summary_rag_pipeline = create_summary_pipeline()


    # mic_prompt_builder = PromptBuilder(template=template)
    # mic_basic_rag_pipeline = Pipeline()
    # mic_basic_rag_pipeline.add_component("text_embedder", text_embedder)
    # mic_basic_rag_pipeline.add_component("retriever", retriever)
    # mic_basic_rag_pipeline.add_component("prompt_builder", prompt_builder)
    # mic_basic_rag_pipeline.add_component("llm", OpenAIGenerator(model="gpt-3.5-turbo"))
    # mic_basic_rag_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
    # basic_rag_pipeline.connect("retriever", "prompt_builder.documents")
    # basic_rag_pipeline.connect("prompt_builder", "llm")


    st.title("👁️‍🗨️🌉 SF Conspiracy Theory Generator ")
    st.write(
        "This is a chatbot powered by OpenAI's GPT-3.5-Turbo, orchestrated by Haystack 2.0 to generate conspiracy theories about the city of San Francisco."
    )

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("What is up?"):
        prompt = temp_mic_prompt
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        result = input_pipeline.run({
            "text_embedder": {"text": prompt},
            "prompt_builder": {"question": prompt}
        })

        documents = result.get("llm")
        if documents:
            with st.chat_message("assistant"):
                response = documents["replies"][0]
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})

            summary_result = summary_rag_pipeline.run({
                "prompt_builder": {"input": response}
            })

            summary_docs = summary_result.get("llm")
            if summary_docs:
                with st.chat_message("assistant"):
                    summary_response = summary_docs["replies"][0]
                    st.markdown(summary_response)

            # Function to generate an image using DALL-E API based on the conspiracy theory
            def generate_image(prompt: str) -> str:
                try:
                    response = openai.images.generate(
                        prompt=prompt, 
                        n=1,
                        size="512x512"
                    )
                    image_url = response.data[0].url
                    return image_url
                except Exception as e:
                    st.error(f"Error generating image: {e}")
                    return None

            # Function to display the image from a URL
            def display_image(image_url: str, file_path: str, save: bool = False):
                try:
                    response = requests.get(image_url)
                    img = Image.open(BytesIO(response.content))

                    if save:
                        if not os.path.exists(file_path):
                            os.makedirs(file_path)
                        img.save(os.path.join(file_path, "generated_image.png"))
                    
                    st.image(img, caption="Generated by DALL-E", use_column_width=True)
                except Exception as e:
                    st.error(f"Error displaying image: {e}")

            image_url = generate_image(summary_response)
            if image_url:
                display_image(image_url, file_path="images", save=True)
            else:
                st.error("Failed to generate image.")
        

        else:
            st.error("No documents found. Please check the pipeline configuration.")

        
        