# 🤖 Tech Documentation RAG Agent

A Retrieval-Augmented Generation (RAG) application that answers questions about technical documentation using local LLMs.

## 📋 Features

- ✅ Multi-document RAG system
- ✅ Web-based UI with Streamlit
- ✅ Real-time document retrieval
- ✅ Local LLM integration (Ollama)
- ✅ Vector embeddings with Chroma
- ✅ Support for multiple documentation sources

## 🔧 Prerequisites

1. **Python 3.11+**
2. **Ollama** installed and running locally
   - Download: https://ollama.ai
   - Required models: `llama3.2` and `mxbai-embed-large`
   
   ```bash
   ollama pull llama3.2
   ollama pull mxbai-embed-large
   ```

## 📦 Installation

1. **Clone/Download the project**

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

   Or with `uv` (if using):
   ```bash
   uv sync
   ```

## 🚀 Running the App

### Option 1: Streamlit Web UI (Recommended)
```bash
uv run --active streamlit run streamlit_app.py
```

The app will open in your browser at `http://localhost:8501`

### Option 2: CLI Version
```bash
uv run python main.py 
```

## 📚 Supported Documentation

Currently indexed:
- Spring Boot 4.0 Release Notes
- Spring Boot 4.0 Migration Guide
- MongoDB 8.0 Release Notes
- MongoDB 8.0 Upgrade Guide

To add more sources, edit the `TECH_DOC_URLS` list in `vector.py`:

```python
TECH_DOC_URLS = [
    "https://your-doc-url-1.com",
    "https://your-doc-url-2.com",
]
```

## 🏗️ Project Structure

```
.
├── streamlit_app.py      # Web UI (Streamlit)
├── main.py              # CLI interface
├── vector.py            # Vector store & document retrieval
├── requirements.txt     # Python dependencies
├── chroma_tech_docs/    # Vector database (auto-created)
└── README.md           # This file
```

## 🔍 How It Works

1. **Document Loading**: Scrapes and chunks documentation from URLs
2. **Embedding**: Converts chunks to embeddings using `mxbai-embed-large`
3. **Vector Storage**: Stores embeddings in Chroma vector database
4. **Retrieval**: Finds relevant documents for user questions
5. **Generation**: Uses `llama3.2` to generate context-aware answers

## ⚙️ Configuration

### Chunk Size
Edit `CHUNK_SIZE` in `vector.py` (default: 1000 chars)
- Larger chunks = more context, fewer documents
- Smaller chunks = specific answers, more documents

### Number of Retrieved Documents
Edit `search_kwargs={"k": 5}` in `vector.py` to retrieve more/fewer documents

### Model Selection
Change `model="llama3.2"` in `streamlit_app.py` or `main.py` for different models

## 🛠️ Troubleshooting

### "Connection refused" error
- Make sure Ollama is running: `ollama serve`
- Check if models are installed: `ollama list`

### "Model not found" error
- Pull the required model: `ollama pull llama3.2`

### Slow response
- Reduce `CHUNK_SIZE` for faster processing
- Reduce `k` value in retriever for fewer documents

### Database issues
- The vector database is auto-cleared on startup
- To reset manually, delete the `chroma_tech_docs/` folder

## 📖 Usage Examples

**Q:** "What are the breaking changes in Spring Boot 4.0?"
```
Retrieved relevant migration guide sections
Generated comprehensive answer with specific changes
```

**Q:** "How do I upgrade MongoDB from 7.0 to 8.0?"
```
Retrieved upgrade guide documentation
Generated step-by-step upgrade instructions
```

## 🔐 Security Note

This application runs entirely locally:
- No data sent to external APIs
- All processing happens on your machine
- Requires local Ollama installation

## 📝 License

MIT

## 🤝 Contributing

Feel free to extend with:
- More documentation sources
- Different LLM models
- Fine-tuned prompts
- Agent-based features


