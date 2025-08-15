from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

load_dotenv()

groq_key = os.getenv("GROQ_API_KEY")

llm = ChatGroq(model="llama3-70b-8192", api_key=groq_key)

messages = [
    SystemMessage(content="You are an expert in social media content startegy."),
    HumanMessage(content="Give a short tip to create engaging content for Instagram."),
]
result = llm.invoke(messages)

print(result.content)