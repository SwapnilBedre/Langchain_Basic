from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda, RunnableParallel, RunnableSequence

# Load environment variables
load_dotenv()

groq_key = os.getenv("GROQ_API_KEY")

model = ChatGroq(model="llama3-70b-8192", api_key=groq_key)

# Define the prompt template
animal_fast_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a fact expert who knows facts about {animal}"),
        ("human", "Tell me {fact_count} facts."),
    ]
)

# Define the prompt template
translation_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You arethe translator so convert the text into {language}"),
        ("human", "translate the following text to {language}: {text}"),
    ]
)

# create a indiviasual runnable lambda
count_words = RunnableLambda(lambda x: f"word count: {len(x.split())}\n{x}")
prepare_for_translation =  RunnableLambda(lambda output: {"text": output, "language": "French"})

# Create a chain.

chain = animal_fast_template | model | StrOutputParser() | prepare_for_translation | translation_template | model | StrOutputParser()

result = chain.invoke({"animal": "Dog", "fact_count": 2})

print(result)