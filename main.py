# import required dependencies
# https://docs.chainlit.io/integrations/langchain
import os
from langchain import hub
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.embeddings import JinaEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import chainlit as cl
from langchain.chains import RetrievalQA

ABS_PATH: str = os.path.dirname(os.path.abspath(__file__))
#DB_DIR: str = os.path.join(ABS_PATH, "medgurus_db_mxbai-embed-large")
#DB_DIR: str = os.path.join(ABS_PATH, "medgurus_db_sammcj__sfr-embedding-mistral:Q4_K_M")
#DB_DIR: str = os.path.join(ABS_PATH, "medgurus_db_codellama__7b-code")
#DB_DIR: str = os.path.join(ABS_PATH, "medgurus_db_snowflake-arctic-embed")
#DB_DIR: str = os.path.join(ABS_PATH, "medgurus_db")
# DB_DIR: str = os.path.join(ABS_PATH, "medgurus_db_jina_base")
# DB_DIR: str = os.path.join(ABS_PATH, "medgurus_db_jina-embeddings-v2-base-code")
# DB_DIR: str = os.path.join(ABS_PATH, "medgurus_db_jina-embeddings-v2-base-ollama")
# DB_DIR: str = os.path.join(ABS_PATH, "medgurus_db_BAAI-bge-base-en-v1.5")
# DB_DIR: str = os.path.join(ABS_PATH, "medgurus_db_jina-embeddings-v2-base-code-local")
# DB_DIR: str = os.path.join(ABS_PATH, "mastodon_db_jina-embeddings-v2-base-code-local")# 
# DB_DIR: str = os.path.join(ABS_PATH, "carbonite_db_jina-embeddings-v2-base-code-local")
DB_DIR: str = os.path.join(ABS_PATH, "phoenix_db_jina-embeddings-v2-base-code-local")

# Set up RetrievelQA model
# rag_prompt_mistral = hub.pull("rlm/rag-prompt-mistral")
# rag_prompt = hub.pull("rlm/rag-prompt")
# print("-------")
# print(repr(rag_prompt))
# print("-------")

from langchain_core.prompts import PromptTemplate

template = """Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Use three sentences maximum and keep the answer as concise as possible.
Always say "thanks for asking!" at the end of the answer.

{context}

Question: {question}

Helpful Answer:"""
rag_prompt = PromptTemplate.from_template(template)

def load_model():
    llm = Ollama(
        model="llama3:8b",
        verbose=True,
        callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
    )
    return llm


def retrieval_qa_chain(llm, vectorstore):
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=vectorstore.as_retriever(
            search_type="similarity",  # Also test "mmr"
            search_kwargs={"k": 10},
        ),
        chain_type_kwargs={"prompt": rag_prompt},
        return_source_documents=True,
    )
    return qa_chain


def qa_bot():
    from langchain_openai import OpenAIEmbeddings
    #persist_directory=DB_PATH, embedding_function=OllamaEmbeddings(model="sammcj/sfr-embedding-mistral:Q4_K_M")
    #persist_directory=DB_PATH, embedding_function=OpenAIEmbeddings(disallowed_special=()
    #persist_directory=DB_PATH, embedding_function=OllamaEmbeddings(model="snowflake-arctic-embed:latest")
    llm = load_model()
    DB_PATH = DB_DIR
    # vectorstore = Chroma(
    #   persist_directory=DB_PATH, embedding_function=OpenAIEmbeddings(disallowed_special=())
    # )
    # vectorstore = Chroma(
    #     persist_directory=DB_PATH, 
    #     embedding_function=JinaEmbeddings(
    #         jina_api_key="jina_3ddc3a9c7010462ba25b20032079740eceQ-7WlOhEcDNsO6ZTHHboc8vPR3", 
    #         model_name="jina-embeddings-v2-base-code"
    #     )
    # )
    # vectorstore = Chroma(
    #     persist_directory=DB_PATH, 
    #     embedding_function=OllamaEmbeddings(model="BAAI-bge-base-en-v1.5:latest")
    # )
    vectorstore = Chroma(
        persist_directory=DB_PATH, 
        embedding_function=HuggingFaceEmbeddings(model_name="jinaai/jina-embeddings-v2-base-code", model_kwargs={'trust_remote_code': True})
    
    )
    
    qa = retrieval_qa_chain(llm, vectorstore)
    return qa


@cl.on_chat_start
async def start():
    """
    Initializes the bot when a new chat starts.

    This asynchronous function creates a new instance of the retrieval QA bot,
    sends a welcome message, and stores the bot instance in the user's session.
    """
    chain = qa_bot()
    welcome_message = cl.Message(content="Starting the bot...")
    await welcome_message.send()
    welcome_message.content = (
        "Hi, Welcome to Chat With Documents using Ollama (mistral model) and LangChain."
    )
    await welcome_message.update()
    cl.user_session.set("chain", chain)


@cl.on_message
async def main(message):
    """
    Processes incoming chat messages.

    This asynchronous function retrieves the QA bot instance from the user's session,
    sets up a callback handler for the bot's response, and executes the bot's
    call method with the given message and callback. The bot's answer and source
    documents are then extracted from the response.
    """
    chain = cl.user_session.get("chain")
    cb = cl.AsyncLangchainCallbackHandler()
    cb.answer_reached = True
    # res=await chain.acall(message, callbacks=[cb])
    res = await chain.acall(message.content, callbacks=[cb])
    # print(f"response: {res}")
    answer = res["result"]
    #answer = answer.replace(".", ".\n")
    source_documents = res["source_documents"]

    text_elements = []  # type: List[cl.Text]

    if source_documents:
        for source_idx, source_doc in enumerate(source_documents):
            source_name = f"{source_doc.metadata['source']}"
            # Create the text element referenced in the message
            text_elements.append(
                cl.Text(content=source_doc.page_content, name=source_name, display="side")
            )
        source_names = [text_el.name for text_el in text_elements]

        if source_names:
            answer += f"\nSources:\n"
            answer += ",\n ".join(source_names)
        else:
            answer += "\nNo sources found"

    await cl.Message(content=answer, elements=text_elements).send()
