import os
from langchain_groq import ChatGroq
from langchain_chroma import Chroma



# defining the derectory containing the text files and the persistant directory
current_dir = os.path.dirname(os.path.abspath(__file__))
persistant_directory = os.path.join(current_dir, "db", "chroma_db")