# filename: vectorstore.py
# from langchain.vectorstores import Milvus #deprecated
from langchain_milvus import Milvus
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pymilvus import connections

from config import milvus_host, milvus_port, collection_name, API_KEY
import os

def get_vectorstore():
    """Initialize and return the Milvus vectorstore connected to Gemini embeddings."""
    # 1) Connect to Milvus
    connections.connect(alias="default", host=milvus_host, port=milvus_port)

    # 2) Init embeddings (Gemini)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=os.environ.get(API_KEY))

    # 3) Load an existing collection as a vectorstore
    try:
        vectorstore = Milvus(
            embedding_function=embeddings,
            collection_name=collection_name,
            connection_args={"host": milvus_host, "port": str(milvus_port)},
            text_field="text",         # <-- important
            metadata_field="metadata", # optional explicit param depending on langchain version
            vector_field="embedding"
        )
    except:
        vectorstore = Milvus(
            embedding_function=embeddings,
            collection_name=collection_name,
            connection_args={"host": milvus_host, "port": str(milvus_port)},
            text_field="text",         # <-- important
            metadata_field="metadata", # optional explicit param depending on langchain version
            embedding_field="embedding"
    )
    return vectorstore