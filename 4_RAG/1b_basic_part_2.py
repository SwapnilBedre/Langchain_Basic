import os
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings



# defining the derectory containing the text files and the persistant directory
current_dir = os.path.dirname(os.path.abspath(__file__))
persistent_directory = os.path.join(current_dir, "db", "chroma_db")

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Use croma DB to store vectors.

db = Chroma(
    persist_directory=persistent_directory,
    embedding_function = embeddings
)

query = "what are the batman's rules?"

# configure the retriver
retriever = db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k":5, "score_threshold":0.5},
)

relevant_docs = retriever.invoke(query)

# Display the relevant results with metadata
print("\n--- Relevant Documents ---")
for i, doc in enumerate(relevant_docs, 1):
    print(f"Document {i}:\n{doc.page_content}\n")
    if doc.metadata:
        print(f"Source: {doc.metadata.get('source', 'Unknown')}\n")

# If no results found, try with basic similarity search
if not relevant_docs:
    print("No documents found with score threshold. Trying basic similarity search...")
    basic_results = db.similarity_search(query, k=3)
    print(f"\n--- Basic Similarity Search Results ({len(basic_results)} found) ---")
    for i, doc in enumerate(basic_results, 1):
        print(f"Document {i}:\n{doc.page_content}\n")
        if doc.metadata:
            print(f"Source: {doc.metadata.get('source', 'Unknown')}\n") 