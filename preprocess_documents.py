import os
import pickle
from pathlib import Path
from haystack import Document
from haystack.components.embedders import SentenceTransformersDocumentEmbedder

# Path to the directory containing the .txt files
folder_path = Path('scraped_articles')

# Path to save the processed data
save_path = 'preprocessed_data.pkl'

# Initialize document embedder
doc_embedder = SentenceTransformersDocumentEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")
doc_embedder.warm_up()

def preprocess_and_save_documents(folder_path: Path, save_path: str):
    docs = []
    for file_path in folder_path.glob('*.txt'):
        with file_path.open('r', encoding='utf-8') as file:
            lines = file.readlines()
            title = lines[0].strip()
            summary = lines[2].strip()
            docs.append(Document(content=summary, meta={"title": title}))
    
    # Embed documents
    docs_with_embeddings = doc_embedder.run(docs)
    
    # Save to file
    with open(save_path, 'wb') as f:
        pickle.dump(docs_with_embeddings, f)
    print(f"Documents and embeddings saved to {save_path}")

preprocess_and_save_documents(folder_path, save_path)