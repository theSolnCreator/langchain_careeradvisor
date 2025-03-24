
import os
from dotenv import load_dotenv
from openai import OpenAI
from langsmith import traceable

from langsmith.wrappers import wrap_openai

load_dotenv()

openai_client = wrap_openai(OpenAI())

# This is the retriever we will use in RAG
# This is mocked out, but it could be anything we want
def retriever(query: str):
    results = ["Harrison worked at Kensho"]
    return results

# This is the end-to-end RAG chain.
# It does a retrieval step then calls OpenAI
@traceable
def rag(question):
    docs = retriever(question)
    system_message = """Answer the users question using only the provided information below:
    
    {docs}""".format(docs="\n".join(docs))
    
    return openai_client.chat.completions.create(
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": question},
        ],
        model="gpt-4o-mini",
    )
    
    
rag("where did harrison work")

