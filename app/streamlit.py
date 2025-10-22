# app.py
import streamlit as st
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.vectorstores import FAISS
from langchain.llms.base import LLM
from typing import Optional, List
import os
import requests
from langchain.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

# Embeddings
embeddings_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Custom LLM Class
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

# Load FAISS Vector Store
VECTOR_STORE_PATH = "faiss_index"
vector_store = FAISS.load_local(
    VECTOR_STORE_PATH,
    embeddings=embeddings_model,
    allow_dangerous_deserialization=True
)

# Prompt Template
condense_question_template = """
Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:
"""

CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(condense_question_template)

qa_prompt_template = """
You are NyayaSathi — an expert legal assistant specialized in the Constitution of Nepal and related laws.
Your goal is to help ordinary Nepalese citizens understand the law clearly, not just quote it.
Always:

Cite the exact Article, Clause, or Section from the Constitution or Acts that supports your explanation. Reference full document name if needed.
If multiple laws apply, list each briefly with citations.
Explain why the law says that and how it applies in simple, concise language.
For broad questions, start with a 1-2 sentence summary, then key points with citations.
Avoid jargon; use plain Nepali-English.
If info insufficient, state clearly.
Be extremely concise — short answers only.

Context:
{context}
Question:
{question}
Answer clearly, respectfully, accurately as a knowledgeable Nepali legal expert.
"""

QA_PROMPT = PromptTemplate(
    template=qa_prompt_template, input_variables=["question", "context"]
)

# RAG QA Chain with Memory
qa = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vector_store.as_retriever(),
    condense_question_prompt=CONDENSE_QUESTION_PROMPT,
    combine_docs_chain_kwargs={"prompt": QA_PROMPT},
)

# Streamlit UI
st.set_page_config(page_title="NyaySathi", page_icon="🇳🇵", layout="centered")

st.title("NyayaSathi 🇳🇵")
st.caption("Your Nepalese legal assistant for laws and Constitution.")

# Session state
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Namaste! I am NyayaSathi. How can I help with Nepal's laws?"}
    ]
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask your question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = qa({"question": prompt, "chat_history": st.session_state.chat_history})
            response = result["answer"]
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.session_state.chat_history.append((prompt, response))

# Clear button at bottom
if st.button("Clear Chat"):
    st.session_state.messages = [
        {"role": "assistant", "content": "Namaste! I am NyayaGPT. How can I help with Nepal's laws?"}
    ]
    st.session_state.chat_history = []
    st.rerun()