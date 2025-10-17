# Remove headers, page nos., etc.
import os
from langchain.document_loaders import PyPDFLoader

RAW_FOLDER = "../data/raw/"
PROCESSED_FOLDER = "../data/processed/"

os.makedirs(PROCESSED_FOLDER, exist_ok=True)

for filename in os.listdir(RAW_FOLDER):
    if filename.endswith(".pdf"):
        loader = PyPDFLoader(os.path.join(RAW_FOLDER, filename))
        docs = loader.load()  # List of Document objects

        # Combine all pages into a single cleaned text
        full_text = "\n".join([doc.page_content for doc in docs])

        # Save as one file per PDF
        save_name = filename.replace(".pdf", ".txt")
        with open(os.path.join(PROCESSED_FOLDER, save_name), "w", encoding="utf-8") as f:
            f.write(full_text)

print("Processed all PDFs without chunking.")
