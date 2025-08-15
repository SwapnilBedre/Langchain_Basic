from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

groq_key = os.getenv("GROQ_API_KEY")

llm = ChatGroq(model="llama3-70b-8192", api_key=groq_key)

template = "write a {tone} email to {company} expressing interest in the {position} position, " \
"mentioning {skills} as a key strength. Keep it to 4 lines"

prompt_template = ChatPromptTemplate.from_template(template)

prompt = prompt_template.invoke({
    "tone": "energetic",
    "company": "Yash",
    "position": "AI Engineer",
    "skills": "Python"
})

result = llm.invoke(prompt)

print(result)

# We can give the messages(system messages and Human Messages)