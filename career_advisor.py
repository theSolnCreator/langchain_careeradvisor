import os
from dotenv import load_dotenv
from fastapi import FastAPI
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# Load API keys
load_dotenv()

app = FastAPI()

# Initialize OpenAI LLM
llm = ChatOpenAI(model="gpt-4o", openai_api_key=os.getenv("OPENAI_API_KEY"))

# Set up memory (stores chat history)
memory = ConversationBufferMemory(memory_key="chat_history")

# Define prompt template
template = """
You are an expert career advisor. You provide personalized advice based on the user's career goals and background.
Chat History: {chat_history}
User: {user_input}
AI:
"""
prompt = PromptTemplate(template=template, input_variables=["chat_history", "user_input"])

# Define LangChain chain
career_chain = LLMChain(llm=llm, prompt=prompt, memory=memory)

@app.get("/chat")
def chat(user_input: str):
    try:
        response = career_chain.invoke({"user_input": user_input})
        return {"response": response["text"]}
    except Exception as e:
        return {"error": str(e)}
