from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

load_dotenv()

groq_key = os.getenv("GROQ_API_KEY")

model = ChatGroq(model="llama3-70b-8192", api_key=groq_key)

chat_history = [] # store in list format

# set an initial system message
SystemMessage(content="You are an helpful AI assistant.")
chat_history.append(SystemMessage) #Here we append the system message to the chat history

# chat loop 
while True:
    query = input("You: ")
    if query.lower() == "exit":
        break
    chat_history.append(HumanMessage(content=query)) #Here we append the user input to the chat history

    # Get AI response using History
    result = model.invoke(chat_history)
    response = result.content
    chat_history.append(AIMessage(content=response)) #Here we append the AI response to the chat history
    
    print(f"AI : {response}") 

print("______Message History______")
print(response.content)