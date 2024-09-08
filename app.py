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
from image_utils import *

def load_app(use_mic_template: bool, text=None):
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


    template = """
    generate a creative and outlandish conspiracy theory in two paragraphs
    Your theory should be based on the news articles about events in San Francisco from the last year, provided below on the Context.
    Make sure the theory is outlandish but still incorporates specific details from news articles that could only be known if you read them.
    Tell the theory as if it is the truth. Don't explicitly say you're pulling from news articles, just do it.
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

    text_template = "Using the prompt below in the Question provided, " + template

    mic_template = """Connect the following conversation indicated in the Question to news 
        articles in the Context provided (infer the speaker names based on the conversation 
        and if you can't infer the names, just call them San Francisco Residents), and """ + template
    
    combined_template = mic_template if use_mic_template else text_template

#  (Your total response MUST be 800 characters or less).
    summary_template = """
    Given the following input with a conspiracy title, description, and sources, create a 
    detailed and vivid scene that can be passed into an AI image generator based on the title and description.
    The scene should include specific visual elements, actions, 
    setting, and mood that reflect the core theme of the conspiracy theory. The output should be 
    structured for optimal image generation.


        **Conspiracy Theory Description**:
        {{input}}

    **Requirements**:
    1. **Setting**: Where is the scene taking place? Provide a vivid description of the location, time of day, weather, or other environmental factors.
    2. **Main Characters/Objects**: In one sentence, describe what or who is in the scene? Include any important figures, objects, or entities (e.g., UFOs, secret agents, strange artifacts).
    3. **Key Actions**: What is happening? Describe any movements, interactions, or behaviors.
    4. **Atmosphere and Mood**: Set the tone of the image (e.g., eerie, mysterious, futuristic) in one sentence.
    5. **Visual Details**: In a maximum of one sentence, focus on visual features such as color palette, textures, and any fine details that make the image compelling and related to the conspiracy.

    **Example Output**:
    A futuristic night-time scene in San Francisco, with a large UFO hovering silently above the Golden Gate Bridge. The bridge is dimly lit by streetlights, casting long shadows on the road. Dark clouds swirl ominously in the sky, as the UFO emits a faint, eerie blue glow. Below, several shadowy figures dressed in black suits and sunglasses are standing near the base of the bridge, staring upwards. The city in the background is barely visible through a thick, fog-like mist. The atmosphere is tense, with cool tones dominating the scene. The lighting is dramatic, with sharp contrasts between light and dark areas.

    Now generate a similar prompt based on the given conspiracy theory and at the end of the prompt, add the sentence 'Make the image photorealistic and eerie.'.
        """


    # Create pipelines dynamically based on the selected template
    def create_pipeline(text_embedder, template):
        prompt_builder = PromptBuilder(template=template)
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

    # Pipeline for creating one sentence summary
    input_pipeline = create_pipeline(text_embedder, combined_template)
    summary_rag_pipeline = create_summary_pipeline()


    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    if not use_mic_template:
        with st.container():
            prompt = st.chat_input("What topics do you want a conspiracy on?", key=use_mic_template)
    else:
        prompt = text

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Run the pipeline to get the conspiracy theory
        result = input_pipeline.run({
            "text_embedder": {"text": prompt},
            "prompt_builder": {"question": prompt}
        })

        documents = result.get("llm")
        if documents:
            response = documents["replies"][0]
            with st.chat_message("assistant"):
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})

            # Generate a summary
            summary_result = summary_rag_pipeline.run({
                "prompt_builder": {"input": response}
            })

            summary_docs = summary_result.get("llm")
            if summary_docs:
                summary_response = summary_docs["replies"][0]
                generate_and_display_image_from_summary(summary_response, save=True)
            else:
                st.error("Failed to generate summary.")
        else:
            st.error("No documents found. Please check the pipeline configuration.")

            
            