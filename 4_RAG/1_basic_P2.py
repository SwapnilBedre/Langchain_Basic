import os
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document  # <-- for creating docs
from langchain_community.vectorstores import FAISS  

# Define the persistent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
faiss_index_path = os.path.join(current_dir, "db", "faiss_index")

# Define the embedding model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2") 

# Load the existing vector store with the embedding function
db = FAISS.load_local(
    faiss_index_path,
    embeddings,   # <-- just pass directly, not embedding_function=
    allow_dangerous_deserialization=True
)


# Define the user's question
query = "Where does Gandalf meet Frodo?"

# Retrieve relevant documents based on the query
retriever = db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 5, "score_threshold": 0.5}, 
)
relevant_docs = retriever.invoke(query)

# Display the relevant results with metadata
print("\n--- Relevant Documents ---")
for i, doc in enumerate(relevant_docs, 1):
    print(f"Document {i}:\n{doc.page_content}\n")
    if doc.metadata:
        print(f"Source: {doc.metadata.get('source', 'Unknown')}\n")