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
   python3 -m venv .venv-rag-time && source .venv-rag-time/bin/activate
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

### Ask Questions

"The file "....ex" is missing a module comment. Can you create one that helps new team members understand what it does, and how it works?"

"Given the following Mission, can you please explain what files I should change, and how I can implement the changes?"

## Enhancements

### Use the advanced script to process the codebase

The script `ingest-code.py` is intended to be easy to understand and to modify. For more control over the codebase processing, you can use the `process_codebase.py` script. 

```bash
usage: process_codebase.py [-h] [-c] [-cd CHUNKS_DIR] [-db CHROMA_DB_DIR] [-oh] [-ed] base_directory

Process subdirectories for chunking and ingestion.

positional arguments:
  base_directory        Base directory to process

options:
  -h, --help            show this help message and exit
  -c, --clean           Clean existing chunks and chroma db before processing
  -cd CHUNKS_DIR, --chunks_dir CHUNKS_DIR
                        If given, chunks are stored into this directory.
  -db CHROMA_DB_DIR, --chroma_db_dir CHROMA_DB_DIR
                        Directory for Chroma DB (default: .rag_time/chroma_db)
  -oh, --omit-headers   Do not add filename header in chunks
  -ed, --empty_db       Only create an empty chroma db

```

To do achieve the same result as `ingest-code.py`, you can run:

```bash
python process_codebase.py -c -db .rag_time/chroma_db ./discourse
```

To use this in chainlit, you can set the `CODEBASE_PATH` environment variable to the directory you want to process:

```bash
export VECTOR_DB_PATH=".rag_time/chroma_db" ; chainlit run main.py
```

The switches `-c`, `-ed` and `-oh` are useful to evaluate the impact of different processing options.