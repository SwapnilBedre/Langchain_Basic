from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_groq import ChatGroq
import os
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS  

# Define the directory containing the text file and the persistent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "documents", "lord_of_the_rings.txt")
faiss_index_path = os.path.join(current_dir, "db", "faiss_index")

# Check if the Chroma vector store already exists
if not os.path.exists(faiss_index_path):
    print("Persistent directory does not exist. Initializing vector store...")

    # Ensure the text file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"The file {file_path} does not exist. Please check the path."
        )

    # Read the text content from the file
    loader = TextLoader(file_path, encoding="utf-8")
    documents = loader.load()

    # Split the document into chunks
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50) 
    docs = text_splitter.split_documents(documents)

    # Display information about the split documents
    print("\n--- Document Chunks Information ---")
    print(f"Number of document chunks: {len(docs)}")
    print(f"Sample chunk:\n{docs[0].page_content}\n")

    # Create embeddings
    print("\n--- Creating embeddings ---")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")  # Update to a valid embedding model if needed
    print("\n--- Finished creating embeddings ---")

    # Create the vector store and persist it automatically
    print("\n--- Creating vector store ---")
    db = FAISS.from_documents(docs, embeddings)
    print("\n--- Finished creating vector store ---")

    # ✅ Save FAISS index
    db.save_local(faiss_index_path)
    print("\n--- Finished creating and saving FAISS vector store ---")

else:
    print("Vector store already exists. No need to initialize.")




# Questions to ask
# Who is the Ring-bearer?
# Where does Gandalf meet Frodo?