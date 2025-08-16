import os
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings



# defining the derectory containing the text files and the persistant directory
current_dir = os.path.dirname(os.path.abspath(__file__))
persistant_directory = os.path.join(current_dir, "db", "chroma_db")

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Use croma DB to store vectors.

db = Chroma(
    persist_directory=persistant_directory,
    embedding_function = embeddings
)

query = "what are the batman's rules?"

# configure the retriver
retriver = db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k":3, "score_threshold":0.5},
)

relevant_docs = retriver.invoke(query)

# Display the relevant results with metadata
print("\n--- Relevant Documents ---")
for i, doc in enumerate(relevant_docs, 1):
    print(f"Document {i}:\n{doc.page_content}\n")
    if doc.metadata:
        print(f"Source: {doc.metadata.get('source', 'Unknown')}\n")