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

from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import LanguageParser
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain.text_splitter import Language

warnings.simplefilter("ignore")

ABS_PATH: str = os.path.dirname(os.path.abspath(__file__))
DB_PATH: str = os.path.join(ABS_PATH, "phoenix_db_jina-embeddings-v2-base-code-local")

language = Language.RUBY
#language = Language.RUBY
code_folder = "./phoenix"
chunk_size = 3000
chunk_overlap = 400

def create_vector_database():
    """
    Creates a vector database using document loaders and embeddings.

    This function loads data from PDF, markdown and text files in the 'data/' directory,
    splits the loaded documents into chunks, transforms them into embeddings using OllamaEmbeddings,
    and finally persists the embeddings into a Chroma vector database.

    """
    dotenv.load_dotenv()

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
        persist_directory=DB_PATH,
    )

    vector_database.persist()
    query_vector_database(vector_database)


def chunk_code():
    parser=LanguageParser(language=language)
    loader = GenericLoader.from_filesystem(
        code_folder,
        glob="**/*",
        suffixes=[".ex", ".exs"],
        parser=parser,
    )
    loaded_documents = loader.load()
    splitter = RecursiveCharacterTextSplitter.from_language(
        language=language, chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    chunked_documents = splitter.split_documents(loaded_documents)
    return chunked_documents


def chunk_docs():
    loader = DirectoryLoader(code_folder, glob="**/*.md", loader_cls=UnstructuredMarkdownLoader)
    loaded_documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunked_documents = splitter.split_documents(loaded_documents)
    return chunked_documents


def query_vector_database(vector_database):
    query = """
    given the following code from `phoenix/lib/phoenix/test/conn_test.ex`, what would I need to change to persist conn.remote_ip in the same way as conn.host?

    ```
        def recycle(conn, headers \\ ~w(accept accept-language authorization)) do
            build_conn()
            |> Map.put(:host, conn.host)
            |> Plug.Test.recycle_cookies(conn)
            |> Plug.Test.put_peer_data(Plug.Conn.get_peer_data(conn))
            |> copy_headers(conn.req_headers, headers)
        end
    ```


    """
    source_documents = vector_database.similarity_search(query, k=10)
    for source_idx, source_doc in enumerate(source_documents):
        print(source_idx)
        print(repr(source_doc.metadata))
        print
        print

if __name__ == "__main__":
    create_vector_database()
