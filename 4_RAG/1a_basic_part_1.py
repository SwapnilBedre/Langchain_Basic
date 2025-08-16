
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_groq import ChatGroq
import os
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader



# defining the directory containing the text files and the persistant directory
current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "documents", "batman_description.txt")
persistent_directory = os.path.join(current_dir, "db", "chroma_db")


# Use croma DB to store vectors.
if not os.path.exists(persistent_directory):
    print("persistant_directory does not exist, initializing vector store...")

    # ensure the text file exists.
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"File not {file_path} does not exist. Please check the file path."
        )
    
    # reload the text content from the file
    # loader = TextLoader("f:/Langchain_Basic/4_RAG/documents/batman_description.txt", encoding="utf-8")
    loader = TextLoader(file_path, encoding="utf-8")
    documents = loader.load()

    # split the text into chunks
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.split_documents(documents)

    #  Display information about the split documents
    print("\n------- Split Documents Information -------")
    print(f"Total number of documents chunks: {len(docs)}")
    print(f"sample chunk: \n{docs[0].page_content}\n")

    # create the embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2")
    print("\n ---- Finished creating embeddings ----")    #Update to void embedding model if needed

    # create the vector store and persist it automatically
    print("\n---- Creating vector store ----")
    db = Chroma.from_documents(
        docs,
        embeddings,
        persist_directory=persistent_directory
    )
    print("Vector store created and persisted successfully.")

else:
    print("Persistant directory already exists, loading vector store...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma(
        persist_directory=persistent_directory,
        embedding_function=embeddings
    )
    print(f"Loaded existing vector store with {db._collection.count()} documents.")
