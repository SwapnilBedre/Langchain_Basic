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
summary_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a movie critic"),
        ("human", "provide a breafsummary of movie {movie_name}"),
    ]
)

# def plot analyze step
def analyze_plot(plot):
    plot_template = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a movie critic"),
            ("human", "analyse the plot: {plot}. what are its strengths and weeknesses?"),
        ]
    )
    return plot_template.format_prompt(plot=plot)


# def character analyze step
def analyze_character(character):
    character_template = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a movie critic"),
            ("human", "analyse the character: {character}. what are its strengths and weeknesses?"),
        ]
    )
    return character_template.format_prompt(character=character)

# Define the prompt template
translation_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You arethe translator so convert the text into {language}"),
        ("human", "translate the following text to {language}: {text}"),
    ]
)

# combine analysis into a final verdict
def combine_verdict(plot_analysis, character_analysis):
    return(f"Plot Analysis:\n{plot_analysis}\n\nCharacter Analysis:\n{character_analysis}")

# simplify branches with LCEL
plot_branch_chain = (
    RunnableLambda(lambda x: analyze_plot(x)) | model | StrOutputParser()
)

character_branch_chain = (
    RunnableLambda(lambda x: analyze_character(x)) | model | StrOutputParser()
)

# Create a combined  chain using langchain expression language (LCEL)

chain = (
    summary_template
    | model
    | StrOutputParser()
    | RunnableParallel(branches={"plot": plot_branch_chain, "character": character_branch_chain})
    | RunnableLambda(lambda x: combine_verdict(x["branches"]["plot"], x["branches"]["character"]))
)

result = chain.invoke({"movie_name": "hulk"})

print(result)