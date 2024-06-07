# rag_time

# langchain-Ollama-Chainlit

Simple Chat UI as well as chat with documents using LLMs with Ollama (mistral model) locally, LangChaiin and Chainlit

In these examples, we’re going to build a simpel chat UI and a chatbot QA app. We’ll learn how to:

#### Example 1

- Create a simple Chat UI locally.

#### Example 2

- Ingest documents into vector database, store locally (creates a knowledge base)
- Create a chainlit app based on that knowledge base.

#### Example 3

- Upload a document(pdf)
- Create vector embeddings from that pdf
- Create a chatbot app with the ability to display sources used to generate an answer

---

### Chat with your documents 🚀

- [Ollama](https://ollama.ai/) and `mistral`as Large Language model
- [LangChain](https://python.langchain.com/en/latest/modules/models/llms/integrations/huggingface_hub.html) as a Framework for LLM
- [Chainlit](https://docs.chainlit.io/) for deploying.

## System Requirements

You must have Python 3.9 or later installed. Earlier versions of python may not compile.

---

## Steps to Replicate

1. Rename example.env to .env with `cp example.env .env`and input the langsmith environment variables. This is optional.

2. Create a virtualenv and activate it

   ```
    && source .venv/bin/activate
   ```

3. Run the following command in the terminal to install necessary python packages:

   ```
   pip install -r requirements.txt
   ```

4. Run the following command in your terminal to start the chat UI:

   ```
   # Example 1
   python3 ingest.py
   chainlit run main.py
   ```

   ***

   ```
   # Example 2
   chainlit run rag.py
   ```

   ***
