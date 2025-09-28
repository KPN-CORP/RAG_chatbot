# filename: vectorstore.py
# from langchain.vectorstores import Milvus #deprecated
from langchain_milvus import Milvus
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pymilvus import connections

from config import HOST, PORT, COL, API_KEY
import os

def get_vectorstore():
    """Initialize and return the Milvus vectorstore connected to Gemini embeddings."""
    # 1) Connect to Milvus
    connections.connect(alias="default", host=HOST, port=PORT)

    # 2) Init embeddings (Gemini)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=os.environ.get(API_KEY))

    # 3) Load an existing collection as a vectorstore
    vectorstore = Milvus(
        embedding_function=embeddings,
        collection_name=COL,
        connection_args={"host": HOST, "port": PORT},
        text_field="metadata",
        vector_field="embedding",
    )
    return vectorstore