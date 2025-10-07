# filename: vectorstore.py
# from langchain.vectorstores import Milvus #deprecated
from langchain_milvus import Milvus
from pymilvus import connections
from model.embbedings import get_gemini_embeddings

from config import (
    milvus_host, milvus_port, collection_name, google_api_key_name, embedding_model
)
import os

def get_vectorstore():
    """Initialize and return the Milvus vectorstore connected to Gemini embeddings."""
    # 1) Connect to Milvus
    connections.connect(alias="default", host=milvus_host, port=milvus_port)

    # 2) Init embeddings (Gemini)
    embeddings = get_gemini_embeddings(embedding_model, os.environ.get(google_api_key_name))

    # 3) Load an existing collection as a vectorstore
    try:
        vectorstore = Milvus(
            embedding_function=embeddings,
            collection_name=collection_name,
            connection_args={"host": milvus_host, "port": str(milvus_port)},
            text_field="text",         # <-- important
            # metadata_field="metadata", # optional explicit param depending on langchain version
            enable_dynamic_field=True,    # new way to handle metadata
            vector_field="embedding"
        )
    except Exception as e:
        print(("Using fallback with embedding_field='embedding' due to error:", e))
        vectorstore = Milvus(
            embedding_function=embeddings,
            collection_name=collection_name,
            connection_args={"host": milvus_host, "port": str(milvus_port)},
            text_field="text",         # <-- important
            # metadata_field="metadata", # optional explicit param depending on langchain version
            enable_dynamic_field=True,    # new way to handle metadata
            embedding_field="embedding"
    )
    return vectorstore