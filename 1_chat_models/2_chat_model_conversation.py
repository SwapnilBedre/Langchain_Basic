from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_groq import ChatGroq
from dotenv import load_dotenv  #to secure and sorted imp files, keys from main code. 
import os   
# (OS - we deal with datasets, models, logs andconfigs, all which are stored in files/folders, to asses it we used os)

load_dotenv()   # to access files in .env file

groq_key = os.getenv("GROQ_API_KEY")

llm = ChatGroq(model="llama3-70b-8192", api_key=groq_key) 

messages = [
    SystemMessage(content="You are an expert in social media content startegy."),
    HumanMessage(content="Give a short tip to create engaging content for Instagram."),
]
result = llm.invoke(messages)  #invoke- function to which we give the input it send it to LLM and return the output.

print(result.content)    #content- is use to convert AI message into a string format. 