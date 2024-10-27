import os
import sys
from langchain_chroma import Chroma 
from langchain_huggingface import HuggingFaceEmbeddings 

def connect_to_chroma(chroma_db_dir):
    
    if not os.path.exists(chroma_db_dir):
        print(f"Error: Chroma database not found in {chroma_db_dir}")
        return
    
    embeddings = HuggingFaceEmbeddings(
        model_name="jinaai/jina-embeddings-v2-base-code",
        model_kwargs={"trust_remote_code": True},
    )
    
    vector_database = Chroma(persist_directory=chroma_db_dir, embedding_function=embeddings)
    return vector_database

def count_chroma_records(vector_database):
    num_records = len(vector_database._collection.get(include=[])['ids'])
    return num_records

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python count_chroma_records.py <directory>")
        sys.exit(1)
    
    directory = sys.argv[1]
    if not os.path.isdir(directory):
        print(f"Error: {directory} is not a valid directory.")
        sys.exit(1)
    
    vector_database = connect_to_chroma(directory)
    print(f"Number of records in Chroma database at {directory}: {count_chroma_records(vector_database)}")
