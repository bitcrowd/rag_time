import os
import warnings
import dotenv

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import LanguageParser
from langchain_community.document_loaders import UnstructuredMarkdownLoader

from helpers import load_env

warnings.simplefilter("ignore")

attrs = load_env()

chunk_size = 3000
chunk_overlap = 400

def create_vector_database():
    """
    Creates a vector database using document loaders and embeddings.

    This function loads data from markdown, text and code files in the codebase directory,
    splits the loaded documents into chunks, transforms them into embeddings using HuggingFaceEmbeddings,
    and finally persists the embeddings into a Chroma vector database.

    """
    chunked_documents = []
    chunked_documents.extend(chunk_code())
    chunked_documents.extend(chunk_docs())

    embeddings = HuggingFaceEmbeddings(
        model_name="jinaai/jina-embeddings-v2-base-code",
        model_kwargs={'trust_remote_code': True}
    )

    vector_database = Chroma.from_documents(
        documents=chunked_documents,
        embedding=embeddings,
        persist_directory=attrs['VECTOR_DB_PATH'],
    )

    vector_database.persist()


def chunk_code():
    parser=LanguageParser(language=attrs['CODEBASE_LANGUAGE'])
    loader = GenericLoader.from_filesystem(
        attrs['CODEBASE_PATH'],
        glob="**/*",
        suffixes=[attrs['CODE_SUFFIX']],
        parser=parser,
    )
    loaded_documents = loader.load()
    splitter = RecursiveCharacterTextSplitter.from_language(
        language=attrs['CODEBASE_LANGUAGE'],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunked_documents = splitter.split_documents(loaded_documents)
    return chunked_documents


def chunk_docs():
    loader = DirectoryLoader(
        attrs['CODEBASE_PATH'],
        glob="**/*.md",
        loader_cls=UnstructuredMarkdownLoader
    )
    loaded_documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunked_documents = splitter.split_documents(loaded_documents)
    return chunked_documents

if __name__ == "__main__":
    create_vector_database()
