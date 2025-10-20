from langchain.prompts import PromptTemplate
prompt_template = """
You are an expert on the Constitution of Nepal.Answer like a lawyer helping a common citizen reagrding laws

Context:
{context}

Question:
{question}

Answer concisely and clearly.
"""
prompt = PromptTemplate(
    input_variables=['context','question'],
    template=prompt_template)