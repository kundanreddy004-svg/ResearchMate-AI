# 🔬 ResearchMate AI
### Local AI-Powered Research Assistant

ResearchMate AI is an AI-powered research assistant designed to help users analyze research papers, retrieve relevant information, conduct web research, and export research summaries as PDF reports.

The application uses locally running language models through Ollama and Retrieval-Augmented Generation (RAG) to answer questions about uploaded research papers.

## 🚀 Features

- **PDF Research Paper Analysis:** Upload and index research papers.
- **Local AI Assistant:** Generate answers using Llama 3.2 through Ollama.
- **Retrieval-Augmented Generation (RAG):** Retrieve relevant information from indexed research papers.
- **Web Research:** Generate research summaries using web sources.
- **Source Citations:** Display sources supporting web research results.
- **PDF Report Export:** Download generated research summaries as PDF files.
- **Research History:** View previously generated research topics.
- **Multiple-Paper Support:** Index multiple research papers and ask questions about their contents.

## 🛠️ Technology Stack

- Python
- Streamlit
- Ollama
- Llama 3.2
- nomic-embed-text
- Retrieval-Augmented Generation (RAG)
- PDF Processing
- Semantic Search
- Web Search Integration

## 🏗️ System Workflow

1. Users upload research papers in PDF format.
2. The application extracts and processes document content.
3. The embedding model converts document chunks into vector representations.
4. Relevant chunks are retrieved when a user asks a question.
5. Llama 3.2 generates answers using the retrieved context.
6. Users can conduct web research and view source references.
7. Research summaries can be exported as PDF reports.

## 💻 Requirements

- Python 3.10 or a compatible version supported by the dependencies
- Ollama
- Llama 3.2 model
- nomic-embed-text embedding model
- Python packages listed in `requirements.txt`
- Internet access for web research

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/kundanreddy004-svg/ResearchMate-AI.git
cd ResearchMate-AI
```

### 2. Create a Virtual Environment

On Windows, open the terminal in VS Code and run:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

```powershell
pip install -r requirements.txt
```

### 4. Download the AI Models

Install Ollama, then run:

```powershell
ollama pull llama3.2
ollama pull nomic-embed-text
```

Ensure Ollama is running before starting the application.

### 5. Start the Application

```powershell
streamlit run app.py
```

Open the local address displayed in the terminal.

## 📄 Example Use Case

A user uploads research papers about artificial intelligence in healthcare and asks questions about their applications, methodologies, findings, and limitations.

ResearchMate AI retrieves relevant document content and generates answers. Users can also conduct web research and export their results as PDF reports.

## 🎯 Project Objectives

- Simplify research paper analysis.
- Enable question answering over academic documents.
- Use local language models for document-based AI assistance.
- Support web-based research and source discovery.
- Generate downloadable research reports.

## 🔮 Future Enhancements

- Dedicated multi-paper comparison.
- Improved citation verification.
- Advanced research analytics.
- Support for additional document formats.
- Improved retrieval and answer accuracy evaluation.

## 👨‍💻 Author

**Kundan Reddy N**

Artificial Intelligence and Data Science  
REVA University

GitHub: https://github.com/kundanreddy004-svg

## ⚠️ Disclaimer

AI-generated answers and research summaries may contain errors. Users should verify important claims against the original research papers and cited sources.