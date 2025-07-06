# vector.py (CLM version)
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
import pandas as pd

df = pd.read_csv("AI_Contract_Intake_Template.csv")
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

db_location = "./clm_langchain_db"
add_documents = not os.path.exists(db_location)

if add_documents:
    documents = []
    ids = []

    for i, row in df.iterrows():
        content = f"""
        Document Name: {row['Document Name']}
        Contract Type: {row['Contract Type']}
        Counterparty Name: {row['Counterparty Name']}
        Effective Date: {row['Effective Date']}
        End Date: {row['End Date']}
        Auto-Renewal Clause: {row['Auto-Renewal Clause']}
        Jurisdiction: {row['Jurisdiction']}
        Contract Value: {row['Contract Value']}
        Confidentiality Clause: {row['Confidentiality Clause']}
        Indemnity Clause: {row['Indemnity Clause']}
        Limitation of Liability Clause: {row['Limitation of Liability Clause']}
        Termination for Convenience: {row['Termination for Convenience']}
        Clause Deviation: {row['Clause Deviation']}
        Risk Level: {row['Risk Level']}
        Recommended Reviewer: {row['Recommended Reviewer']}
        Comments: {row['Comments']}
        """
        document = Document(
            page_content=content,
            metadata={"Contract Type": row["Contract Type"], "Risk Level": row["Risk Level"]},
            id=str(i)
        )
        documents.append(document)
        ids.append(str(i))

vector_store = Chroma(
    collection_name="clm_documents",
    persist_directory=db_location,
    embedding_function=embeddings
)

if add_documents:
    vector_store.add_documents(documents=documents, ids=ids)

retriever = vector_store.as_retriever(search_kwargs={"k": 5})
