from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
from google.cloud import firestore
from langchain_google_firestore import FirestoreChatMessageHistory


# Load environment variables
load_dotenv()

# Setup firebase firestore
PROJECT_ID = "langchain-7c299"
SESSION_ID = "user_session_new"
COLLECTION_NAME = "chat_history"

# Initialize firestore client
print
client = firestore.Client(project=PROJECT_ID)

#  Initialize firestore chat message history
print("Initializing Firestore message history...")
chat_history = FirestoreChatMessageHistory(
    client=client,
    session_id=SESSION_ID,
    collection_name=COLLECTION_NAME
)

print("Firestore message history initialized.")
print("current chat history:", chat_history.messages)

# Initialize the chat model
groq_key = os.getenv("GROQ_API_KEY")

model = ChatGroq(model="llama3-70b-8192", api_key=groq_key)

print("Start chatting with Ai. Type 'exit' to quit")

while True:
    human_input = input("user: ")
    if human_input.lower == "exit":
        break

    chat_history.add_user_message(human_input)

    ai_response = model.invoke(chat_history.messages)
    chat_history.add_ai_message(ai_response.content)

    print(f"AI: {ai_response.content}")