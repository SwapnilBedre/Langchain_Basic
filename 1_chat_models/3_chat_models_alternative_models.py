# from langchain_google_genai import ChatGoogleGenrativeAI
# from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# Set up environment variables
load_dotenv()

messages = [
    SystemMessage(content="You are an expert in social media content startegy."),
    HumanMessage(content="Give a short tip to create engaging content for Instagram."),
]

# Initialize the chat model 

# _________Langchain Groq chat Model_________

groq_key = os.getenv("GROQ_API_KEY")

model = ChatGroq(model="llama3-70b-8192", api_key=groq_key)


result = model.invoke(messages)

print("Answer for Groq: {result.content}")  # Output the content of the AI's response

# _________Langchain Anthropic chat Model_________

# groq_key = os.getenv("GROQ_API_KEY")

# model = ChatGroq(model="llama3-70b-8192", api_key=groq_key)


# result = model.invoke(messages)

# print("Answer for Groq: {result.content}")  # Output the content of the AI's response


# _________Langchain OpenAi chat Model_________Not Done

# openai_key = os.getenv("OPENAI_API_KEY")

# model = ChatOpenAI(model="gpt-4o", api_key=openai_key)

# result = model.invoke(messages)

# print("Answer for Groq: {result.content}")  # Output the content of the AI's response


# _________Langchain Gemini chat Model_________Not Done

# groq_key = os.getenv("GROQ_API_KEY")    

# model = ChatGoogleGenrativeAI(model="gemini-1.5-pro", api_key=groq_key)

# result = model.invoke(messages)

# print("Answer for Groq: {result.content}")  # Output the content of the AI's responsemodel