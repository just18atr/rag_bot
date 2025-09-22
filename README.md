
# RAG-Based Research Bot with LangChain & Gemini

This project implements a **Retrieval-Augmented Generation (RAG) research assistant** that combines **LangChain’s RetrievalQA** with **Gemini** for delivering accurate, context-aware answers. The system enables interactive academic research with document ingestion, semantic search, and citation-backed responses.

---

## 🔑 Features

* **RAG-Powered Question Answering**

  * Combines **LangChain RetrievalQA** with **Gemini** to generate context-aware and reliable answers.
* **Document Ingestion**

  * Supports **PDFs (PyPDFLoader)** and **text files (TextLoader)** with recursive chunking for efficient retrieval.
* **Persistent Vector Storage**

  * Uses **ChromaDB** with **Gemini Embeddings** for semantic search and session-level persistence.
* **Interactive UI**

  * Built using **Streamlit**, allowing users to upload research papers, query them, and receive **citation-backed responses**.

---

## 🛠️ Technology & Tools

* **Frameworks**: LangChain, Streamlit
* **LLM**: Gemini
* **Vector Database**: ChromaDB
* **Embeddings**: Gemini Embeddings
* **Document Loaders**: PyPDFLoader, TextLoader

---

## 📊 Results

* Delivered accurate and **citation-supported answers** across multiple research papers.
* Achieved efficient document retrieval through **recursive text chunking**.
* Enabled persistent semantic search across sessions using **ChromaDB**.

---

## 📂 Repository Structure

```
├── /src             # Core implementation files
├── /docs            # Documentation and screenshots
├── /notebooks       # Prototyping and experiments
└── README.md        # Project description
```

---

## 🚀 Applications

* Academic research assistance
* Literature survey automation
* Intelligent knowledge retrieval from documents

---

