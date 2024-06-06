import os
import warnings
import dotenv

from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.embeddings import JinaEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import DirectoryLoader

from langchain_community.vectorstores import Chroma

from langchain.document_loaders.generic import GenericLoader
from langchain.document_loaders.parsers import LanguageParser
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain.text_splitter import Language

warnings.simplefilter("ignore")

ABS_PATH: str = os.path.dirname(os.path.abspath(__file__))
# DB_DIR: str = os.path.join(ABS_PATH, "medgurus_db")
# DB_DIR: str = os.path.join(ABS_PATH, "medgurus_db_nomic")
# DB_DIR: str = os.path.join(ABS_PATH, "medgurus_db_jina_base")
# DB_DIR: str = os.path.join(ABS_PATH, "medgurus_db_jina-embeddings-v2-base-code")
# DB_DIR: str = os.path.join(ABS_PATH, "medgurus_db_my-jina-latest")
# DB_DIR: str = os.path.join(ABS_PATH, "medgurus_db_BAAI-bge-base-en-v1.5")
# DB_DIR: str = os.path.join(ABS_PATH, "carbonite_db_jina-embeddings-v2-base-code-local")
DB_DIR: str = os.path.join(ABS_PATH, "phoenix_db_jina-embeddings-v2-base-code-local")


# Create vector database
def create_vector_database():
    """
    Creates a vector database using document loaders and embeddings.

    This function loads data from PDF, markdown and text files in the 'data/' directory,
    splits the loaded documents into chunks, transforms them into embeddings using OllamaEmbeddings,
    and finally persists the embeddings into a Chroma vector database.

    """
    # Initialize loaders for different file types
    # pdf_loader = DirectoryLoader("data/", glob="**/*.pdf", loader_cls=PyPDFLoader)
    # loaded_documents = pdf_loader.load()
    # len(loaded_documents)

    dotenv.load_dotenv()

    loader = GenericLoader.from_filesystem(
        "./phoenix",
        glob="**/*",
        suffixes=[".ex", ".exs"],
        parser=LanguageParser(language=Language.RUBY, parser_threshold=500),
    )
    loaded_documents = loader.load()

    ruby_splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.RUBY, chunk_size=3000, chunk_overlap=400
    )
    chunked_documents = ruby_splitter.split_documents(loaded_documents)

    md_loader = DirectoryLoader("phoenix/", glob="**/*.md", loader_cls=UnstructuredMarkdownLoader)
    loaded_md_documents = md_loader.load()
    #len(loaded_documents)

    # Split loaded documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=400)
    chunked_documents.extend(text_splitter.split_documents(loaded_md_documents))

    # Initialize Ollama Embeddings
    # embeddings  = OllamaEmbeddings(model="snowflake-arctic-embed:latest")
    # embeddings  = OllamaEmbeddings(model="codellama:7b-code")
    # embeddings = OpenAIEmbeddings(disallowed_special=())
    # embeddings  = OllamaEmbeddings(model="nomic-embed-text:latest")
    # embeddings = JinaEmbeddings(
    #     jina_api_key="jina_1a0a922322a3475794b23d8fa28dd1ddifoDqIVARQooCC2BleVgiQWBkVti", 
    #     model_name="jina-embeddings-v2-base-code"
    # )
    # embeddings  = OllamaEmbeddings(model="BAAI-bge-base-en-v1.5:latest")

    embeddings = HuggingFaceEmbeddings(model_name="jinaai/jina-embeddings-v2-base-code", model_kwargs={'trust_remote_code': True})

    #Create and persist a Chroma vector database from the chunked documents
    # vector_database = Chroma.from_documents(
    #     documents=chunked_documents,
    #     embedding=open_ai_embeddings,
    #     persist_directory=DB_DIR,
    # )

    vector_database = Chroma.from_documents(
        documents=chunked_documents,
        embedding=embeddings,
        persist_directory=DB_DIR,
    )

    vector_database.persist()
    
    # query it
    query = "What is administrate"
    source_documents = vector_database.similarity_search(query)
    for source_idx, source_doc in enumerate(source_documents):
        print(len(source_doc.page_content))
        print



if __name__ == "__main__":
    create_vector_database()
