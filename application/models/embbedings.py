from langchain_huggingface import HuggingFaceEmbeddings
from config import google_api_key_name
import os

# 1. Initialize the Hugging Face Embedding Model for Indonesian
def get_huggingface_embeddings(semantic_embedding_model):
    """Initializes and returns a Hugging Face embedding model for Indonesian."""
    model_name = semantic_embedding_model
    model_kwargs = {'device': 'cpu'}  # Use 'cuda' for GPU
    encode_kwargs = {'normalize_embeddings': True}
    
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    return embeddings

def get_gemini_embeddings(embedding_model, api_key):
    api_key_eff = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or os.environ.get(google_api_key_name)
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model=embedding_model, google_api_key=api_key_eff)  # type: ignore
    except TypeError:
        embeddings = GoogleGenerativeAIEmbeddings(model=embedding_model, api_key=api_key_eff)  # type: ignore
    
    return embeddings
    