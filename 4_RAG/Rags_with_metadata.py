from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_groq import ChatGroq
import os
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS  

# Define the directory containing the text file and the persistent directory

current_dir = os.path.dirname(os.path.abspath(__file__))
books_dir = os.path.join(current_dir, "documents")
db_dir = os.path.join(current_dir, "db")
faiss_index_path = os.path.join(db_dir, "faiss_db_with_metadata")

print(f"Books Directory: {books_dir}")
print(f"faiss_index_path: {faiss_index_path}")

# Check if the Chroma vector store already exists
if not os.path.exists(faiss_index_path):
    print("faiss_index_path directory does not exist. Initializing vector store...")

    # Ensure the text file exists
    if not os.path.exists(books_dir):
        raise FileNotFoundError(
            f"The file {books_dir} does not exist. Please check the path."
        )
    
    # list all text files in directory
    book_files = [f for f in os.listdir(books_dir) if f.endswith(".txt")]

    # Read the text content from the each file and store with metadata
    documents=[]
    for book_file in book_files:
        file_path= os.path.join(books_dir, book_file)
        loader = TextLoader(file_path, encoding="utf-8")
        documents.extend(loader.load())
        for doc in documents:
            doc.metadata = {"source": book_file}
            documents.append(doc)


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
    db = FAISS.from_documents(docs, embeddings, faiss_index_path)
    print("\n--- Finished creating vector store ---")

    # ✅ Save FAISS index
    db.save_local(faiss_index_path)
    print("\n--- Finished creating and saving FAISS vector store ---")

else:
    print("Vector store already exists. No need to initialize.")




# Questions to ask
# Who is the Ring-bearer?
# Where does Gandalf meet Frodo?

# # Define the embedding model
# embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2") 

# # Load the existing vector store with the embedding function
# db = FAISS.load_local(
#     faiss_index_path,
#     embeddings,   # <-- just pass directly, not embedding_function=
#     allow_dangerous_deserialization=True
# )


# # Define the user's question
# query = "Where does Gandalf meet Frodo?"

# # Retrieve relevant documents based on the query
# retriever = db.as_retriever(
#     search_type="similarity_score_threshold",
#     search_kwargs={"k": 5, "score_threshold": 0.5}, 
# )
# relevant_docs = retriever.invoke(query)

# # Display the relevant results with metadata
# print("\n--- Relevant Documents ---")
# for i, doc in enumerate(relevant_docs, 1):
#     print(f"Document {i}:\n{doc.page_content}\n")
#     if doc.metadata:
#         print(f"Source: {doc.metadata.get('source', 'Unknown')}\n")