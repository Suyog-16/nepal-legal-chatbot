from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from ingest_data import load_data

def build_vectordb():
    embeddings_model = HuggingFaceEmbeddings(model_name = "all-MiniLM-L6-v2")
    chunks = load_data()
    embeddings = [embeddings_model.embed_documents(chunk.page_content) for chunk in chunks]
    vector_store = FAISS.from_documents(chunks,embeddings_model)
    vector_store.save_local("../faiss_index")
