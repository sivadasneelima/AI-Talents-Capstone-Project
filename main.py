from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever

model = OllamaLLM(model="llama3.2")

template = """
You are an expert contract review assistant specialized in Contract Lifecycle Management (CLM).

Your tasks include:
1. Classifying document types (e.g., NDA, MSA, PO).
2. Extracting key metadata (dates, jurisdiction, contract value).
3. Detecting clauses and comparing them to standard policy.
4. Flagging risks (missing clauses, high values, etc.).
5. Identifying timeline triggers (renewals, expirations).
6. Highlighting non-compliance with templates or policies.

Here are some contract excerpts: {reviews}

Here is the reviewer’s question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

while True:
    print("\n\n-------------------------------")
    question = input("Ask your contract question (q to quit): ")
    print("\n\n")
    if question == "q":
        break
    
    reviews = retriever.invoke(question)
    result = chain.invoke({"reviews": reviews, "question": question})
    print(result)