# Standard library imports
import argparse
import glob
import json
import math
import os
import shutil
import sys
import warnings
from itertools import tee
import textwrap  # Add this to imports at top

# Third-party imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    DirectoryLoader,
    TextLoader,
    UnstructuredMarkdownLoader
)
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import LanguageParser
from langchain_chroma import Chroma  # Updated import
from langchain_huggingface import HuggingFaceEmbeddings

# Local imports
from count_chroma_records import count_chroma_records
from helpers import load_env

# Suppress warnings
warnings.simplefilter("ignore")

# Load environment variables
attrs = load_env()

# Get chunk size and overlap from environment variables
CHUNK_SIZE = int(attrs.get('CHUNK_SIZE', 3500))
CHUNK_OVERLAP = int(attrs.get('CHUNK_OVERLAP', 875))

def chunk_code(codebase_path: str, code_suffixes: list, 
               codebase_language: str, omit_headers: bool) -> list:
    """
    Chunks code from a specified codebase path using a language parser.

    Args:
        codebase_path (str): The path to the codebase directory.
        code_suffixes (list): A list of file extensions to be considered as code files.
        codebase_language (str): The language of the codebase.
        omit_headers (bool): Whether to omit file headers in chunks.

    Returns:
        list: A list of chunked documents.
    """
    parser = LanguageParser(language=codebase_language)
    loader = GenericLoader.from_filesystem(
        codebase_path,
        suffixes=code_suffixes,
        parser=parser,
        show_progress=True,
    )
    splitter = RecursiveCharacterTextSplitter.from_language(
        language=codebase_language,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    loaded_documents = loader.lazy_load()
    
    loaded_documents = [doc for doc in loaded_documents 
                       if doc.metadata.get('content_type') != 'simplified_code']
    
    chunked_documents = splitter.split_documents(loaded_documents)
    for chunk in chunked_documents:
        chunk.metadata['source'] = os.path.relpath(chunk.metadata['source'], codebase_path)
        
    if not omit_headers:
        for chunk in chunked_documents:
            chunk.page_content = file_header(chunk.metadata['source']) + chunk.page_content

    return chunked_documents

def file_header(filepath):
    header = """
        File: {filename}
        Path: {filepath}
        
    """
    header = textwrap.dedent(header).format(
        filename=os.path.basename(filepath), 
        filepath=filepath
    )
    return header

def cleanup_directories(chunks_dir: str, chroma_db_dir: str) -> None:
    """
    Clean up the chunks and chroma database directories.

    Args:
        chunks_dir (str): Directory containing chunks
        chroma_db_dir (str): Directory containing the Chroma database
    """
    if chunks_dir is not None and os.path.exists(chunks_dir):
        shutil.rmtree(chunks_dir)
        print(f"Removed {chunks_dir}")

    if os.path.exists(chroma_db_dir):
        shutil.rmtree(chroma_db_dir)
        print(f"Removed {chroma_db_dir}")
    else:
        print(f"No chroma db directory found in {chroma_db_dir}")


def chunk_text_docs(directory):
    loader = DirectoryLoader(
        directory,
        glob=["**/*.yml", "**/*.txt"],
        show_progress=True,
        loader_cls=TextLoader,
    )
    loaded_documents = loader.load()
    for doc in loaded_documents:
        doc.metadata['language'] = 'text'

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    chunked_documents = splitter.split_documents(loaded_documents)
    return chunked_documents

def chunk_markdown_docs(directory):
    loader = DirectoryLoader(
        directory, 
        glob=["**/*.md", "**/*.markdown"], 
        show_progress=True,
        loader_cls=UnstructuredMarkdownLoader,
        loader_kwargs={
            "mode": "single"
        }
    )
    loaded_documents = loader.load()
    for doc in loaded_documents:
        doc.metadata['language'] = 'md'

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    chunked_documents = splitter.split_documents(loaded_documents)
    return chunked_documents

def chunk_code_files(directory, language_config, omit_headers):
    code_chunks = []
    for lang, extensions in language_config.items():
        lang_chunks = chunk_code(directory, extensions, lang, omit_headers)
        code_chunks.extend(lang_chunks)
        print(f"Processed {len(lang_chunks)} chunks for {lang}")
    return code_chunks

def chunk_codebase(directory: str, chunks_dir: str, omit_headers: bool) -> list:
    """
    Process and chunk an entire codebase.

    Args:
        directory (str): Directory containing the codebase
        chunks_dir (str): Directory to save chunks
        omit_headers (bool): Whether to omit file headers in chunks

    Returns:
        list: List of processed chunks
    """
    LANGUAGE_CONFIG = {
        "ruby":   ['.erb', '.rb', '.haml'],
        "js":     ['.jsx', '.js', 'json'],
        "ts":     ['.ts'],
        "elixir": ['.ex', '.exs', '.heex']
    }

    chunks = chunk_markdown_docs(directory)
    print(f"Processed {len(chunks)} chunks for markdown")

    text_chunks = chunk_text_docs(directory)
    print(f"Processed {len(text_chunks)} chunks for text")
    chunks.extend(text_chunks)

    code_chunks = chunk_code_files(directory, LANGUAGE_CONFIG, omit_headers)
    print(f"Processed {len(code_chunks)} chunks for code")
    chunks.extend(code_chunks)

    if chunks_dir:
        save_chunks(chunks, chunks_dir)

    print(f"Processed a total of {len(chunks)} chunks for directory: {directory}")
    return chunks

def save_chunks(chunks, chunks_dir):
    if not os.path.exists(chunks_dir):
        os.makedirs(chunks_dir)

    num_digits = math.ceil(math.log10(len(chunks) + 1))
    
    for index, chunk in enumerate(chunks, start=1):
        file_base_name = os.path.basename(chunk.metadata['source'])
        chunk_file_name = f"chunk_{index:0{num_digits}d}_{file_base_name}.json"
        chunk_file_path = os.path.join(chunks_dir, chunk_file_name)

        chunk_data = {
            "metadata": chunk.metadata,
            "content": chunk.page_content
        }

        with open(chunk_file_path, 'w', encoding='utf-8') as chunk_file:
            json.dump(chunk_data, chunk_file, indent=2)

def store_embeddings(chroma_db_dir: str, chunks: list) -> str:
    """
    Store document embeddings in Chroma database.

    Args:
        chroma_db_dir (str): Directory for Chroma database
        chunks (list): List of document chunks to store

    Returns:
        str: Status message
    """
    embeddings = HuggingFaceEmbeddings(
        model_name="jinaai/jina-embeddings-v2-base-code",
        model_kwargs={"trust_remote_code": True},
    )

    vector_database = Chroma(
        persist_directory=chroma_db_dir, 
        embedding_function=embeddings
    )
    print(f"Loaded {count_chroma_records(vector_database)} existing records from {chroma_db_dir}")
    
    if chunks:  
        vector_database.add_documents(chunks)

    print(f"Now {count_chroma_records(vector_database)} records in {chroma_db_dir}")
    return "embeddings created"

def process_codebase(directory, chunks_dir, chroma_db_dir, clean, empty_db, omit_headers):
    if clean:
        cleanup_directories(chunks_dir, chroma_db_dir)
    
    if empty_db:
        chunks = []
    else:
        chunks = chunk_codebase(directory, chunks_dir, omit_headers)
    
    store_embeddings(chroma_db_dir, chunks)

    if chunks_dir is not None:
        save_chunks(chunks, chunks_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process subdirectories for chunking and ingestion.")
    parser.add_argument("base_directory", help="Base directory to process")
    parser.add_argument("-c", "--clean", action="store_true", help="Clean existing chunks and chroma db before processing")
    parser.add_argument("-cd", "--chunks_dir", default=None, help="If given, chunks are stored into this directory.")
    parser.add_argument("-db", "--chroma_db_dir", default=".rag_time/chroma_db", help="Directory for Chroma DB (default: .rag_time/chroma_db)")
    parser.add_argument("-oh", "--omit-headers", action="store_true", help="Don't add filename in chunks")
    parser.add_argument("-ed", "--empty_db", action="store_true", help="Only create an empty chroma db")
    args = parser.parse_args()

    process_codebase(args.base_directory, args.chunks_dir, args.chroma_db_dir, args.clean, args.empty_db, args.omit_headers)
