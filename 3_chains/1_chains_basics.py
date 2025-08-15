from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser

# Load environment variables
load_dotenv()

groq_key = os.getenv("GROQ_API_KEY")

model = ChatGroq(model="llama3-70b-8192", api_key=groq_key)

# Define the prompt template
prompt_template = ChatPromptTemplate.from_messages([
        ("system", "You are a fact expert who knows facts about {animal}"),
        ("human", "Tell me {fact_count} facts."),
    ]
)

chain = prompt_template | model | StrOutputParser()

result = chain.invoke({"animal": "cat", "fact_count": 3})

print(result)