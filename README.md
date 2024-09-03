# rag_time

Simple Chat UI to chat about a private codebase using LLMs locally.

**Technology:**

- [Ollama](https://ollama.ai/) and [llama3:8b](https://ollama.com/library/llama3:8b) as Large Language Model
- [jina-embeddings-v2-base-code](https://jina.ai/news/elevate-your-code-search-with-new-jina-code-embeddings) as Embedding Model
- [LangChain](https://python.langchain.com/en/latest/modules/models/llms/integrations/huggingface_hub.html) as a framework for LLM
- [Chainlit](https://docs.chainlit.io/) for the chat UI

## Getting started

### Prerequisites

1. Make sure you have Python 3.9 or later installed
2. Download and install [Ollama](https://ollama.com/download)
3. Pull the model:

   ```bash
   ollama pull llama3:8b
   ```

4. Run the model:

   ```bash
   ollama run llama3:8b
   ```

### Run the Chat bot

1. Create a Python virtual environment and activate it:

   ```bash
   python3 -m venv .venv && source .venv/bin/activate
   ```

2. Install Python dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Clone an example repository to question the chat bot about:

   ```bash
   git clone https://github.com/discourse/discourse
   ```

4. Set up the vector database:

   ```bash
   python ingest-code.py
   ```

5. Start the chat bot:

   ```bash
   chainlit run main.py
   ```

6. To exit the Python virtual environment after you are done, run:

   ```bash
   deactivate
   ```

## Make it your own

Modify the `.env` file to run the chat bot on your codebase and language.
