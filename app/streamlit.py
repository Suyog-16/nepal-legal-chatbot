# app.py
import streamlit as st
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.vectorstores import FAISS
from langchain.llms.base import LLM
from typing import Optional, List
import os
import requests
from langchain.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

# -----------------------------
# Embeddings
# -----------------------------
embeddings_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# -----------------------------
# Custom LLM Class
# -----------------------------
class HuggingFaceAPILLM(LLM):
    api_url: str = "https://router.huggingface.co/v1/chat/completions"
    api_key: str = os.environ["HF_TOKEN"]
    model: str = "meta-llama/Meta-Llama-3-8B-Instruct:novita"

    @property
    def _llm_type(self) -> str:
        return "huggingface_api"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}]
        }
        response = requests.post(self.api_url, headers=headers, json=payload)
        result = response.json()
        return result["choices"][0]["message"]["content"]

llm = HuggingFaceAPILLM()

# -----------------------------
# Load FAISS Vector Store
# -----------------------------
VECTOR_STORE_PATH = "faiss_index"

vector_store = FAISS.load_local(
    VECTOR_STORE_PATH,
    embeddings=embeddings_model,
    allow_dangerous_deserialization=True
)

# -----------------------------
# Prompt Template
# -----------------------------
prompt_template = """
You are NyayaGPT — an expert legal assistant specialized in the Constitution of Nepal and related laws.

Your goal is to help ordinary Nepalese citizens **understand** the law clearly, not just quote it.
Whenever possible:
- Cite the exact **Article**, **Clause**, or **Section** from the Constitution or Acts that supports your explanation.
- If multiple laws apply, mention each briefly.
- Explain *why* the law says that, and how it applies to the user's question in simple language.
- If the question is broad, give a short summary first, then key legal points.
- Avoid legal jargon unless needed — use plain Nepali-English explanations where suitable.
- If the context doesn't have enough information, say so clearly.

Context:
{context}

Question:
{question}

Your answer should follow this structure:
1. **Summary:** A one-line summary of the main legal idea.
2. **Legal Reference:** Cite specific Article(s), Clause(s), or Act(s).
3. **Explanation:** Explain the meaning in simple, educational terms.
4. **Practical Example (optional):** If helpful, give a real-life example.

Answer clearly, respectfully, and accurately as a knowledgeable Nepali legal expert.
"""

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=prompt_template
)

# -----------------------------
# RAG QA Chain
# -----------------------------
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vector_store.as_retriever(),
    chain_type_kwargs={"prompt": prompt}
)

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="NyayaGPT", page_icon="🇳🇵", layout="wide")

st.title("NyayaGPT🇳🇵 ")
st.write("Your Nepalese legal assistant. Ask questions about laws, policies, and the Constitution.")

# Sidebar
st.sidebar.title("NyayaGPT Options")
start_fresh = st.sidebar.button("Start Fresh Conversation")

# Session state
if "history" not in st.session_state or start_fresh:
    st.session_state.history = []
    # Preloaded greeting dialogue
    st.session_state.history.append(
        ("NyayaGPT", "Namaste! I am NyayaGPT, your legal assistant. I can help you understand Nepal's Constitution, laws, and policies. How can I help you today?")
    )

# Chat input
query = st.text_input("Ask your question here...", key="query_input")

if query:
    with st.spinner("Fetching answer..."):
        answer = qa.run(query)
        st.session_state.history.append((query, answer))

# Display chat
for q, a in st.session_state.history:
    if q == "NyayaGPT":  # Greeting
        st.markdown(f"**{q}:** {a}")
    else:
        st.markdown(f"**You:** {q}")
        st.markdown(f"**NyayaGPT:** {a}")
    st.markdown("---")
