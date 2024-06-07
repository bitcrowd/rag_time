# rag_time

Simple Chat UI to chat about a private codebase using LLMs locally.

**Technology:**

- [Ollama](https://ollama.ai/) and [llama3:8b](https://ollama.com/library/llama3:8b) as Large Language model
- [LangChain](https://python.langchain.com/en/latest/modules/models/llms/integrations/huggingface_hub.html) as a framework for LLM
- [Chainlit](https://docs.chainlit.io/) for the chat UI

## Getting started

### Prerequisites

1. Make sure you have Python 3.9 or later installed
2. Download and install [Ollama](https://ollama.com/download)
3. Pull the model:

   ```bash
   ollama pull llama:3b
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

3. Set up the vector database:

   ```bash
   python ingest-code.py
   ```

4. Start the chat bot:

   ```bash
   chainlit run main.py
   ```

---

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

---

## St
