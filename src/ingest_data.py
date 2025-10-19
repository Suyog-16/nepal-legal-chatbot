from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
import os
import re
from langchain.schema import Document
from dotenv import load_dotenv
load_dotenv()

def load_data():
    loader = PyPDFLoader(os.environ["RAW_DATA_PATH"])
    pages = loader.load()

    # Merge pages into one long text
    full_text = " ".join([p.page_content.replace("\n", " ").strip() for p in pages])


    # -----------------Split at "Article <number>" patterns-------------------------------------
    chunks = re.split(r"(Article\s+\d+[A-Za-z]?(?:\s*[-–]\s*[^\n]*)?)", full_text)
    documents = []

    for i in range(1, len(chunks), 2):
        title = chunks[i].strip()
        body = chunks[i+1].strip()
        documents.append(Document(
            page_content=body,
            metadata={"article_title": title}
        ))
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 800,
        chunk_overlap= 100
    )
    chunks = text_splitter.split_documents(documents)
    return chunks
