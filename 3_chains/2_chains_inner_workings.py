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
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a fact expert who knows facts about {animal}"),
        ("human", "Tell me {fact_count} facts."),
    ]
)

# create a indiviasual runnable lambda
format_prompt = RunnableLambda(lambda x: prompt_template.format_prompt(**x))
invoke_model = RunnableLambda(lambda x: model.invoke(x.to_messages()))
parse_output = RunnableLambda(lambda x: x.content)

# create a chain using RunnableSequence
chain = RunnableSequence(first=format_prompt, middle=[invoke_model], last=parse_output)

# chain = RunnableSequence(format_prompt | invoke_model | parse_output) --Langchain Expression Language

result = chain.invoke({"animal": "Dog", "fact_count": 2})

print(result)