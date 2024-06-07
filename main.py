# import required dependencies
# https://docs.chainlit.io/integrations/langchain
import os
import warnings

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
import chainlit as cl

from helpers import load_env

warnings.simplefilter("ignore")

attrs = load_env()

template = """Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Use three sentences maximum and keep the answer as concise as possible.
Always say "thanks for asking!" at the end of the answer.

{context}

Question: {question}

Helpful Answer:"""

prompt = PromptTemplate.from_template(template)

def load_model():
    return Ollama(
        model=attrs['OLLAMA_MODEL'],
        verbose=True,
        callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
    )

def load_vector_db():
    return Chroma(
        persist_directory=attrs['VECTOR_DB_PATH'],
        embedding_function=HuggingFaceEmbeddings(
            model_name="jinaai/jina-embeddings-v2-base-code",
            model_kwargs={'trust_remote_code': True}
        )
    )

def qa_chain(llm, vectorstore):
    return RetrievalQA.from_chain_type(
        llm,
        retriever=vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 10},
        ),
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True,
    )

def qa_bot():
    llm = load_model()
    vectorstore = load_vector_db()
    return qa_chain(llm, vectorstore)


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
        "Hi, Welcome to Granny RAG. Ask me anything about your code!"
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
    res = await chain.acall(message.content, callbacks=[cb])
    answer = res["result"]
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
