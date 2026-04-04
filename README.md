# 🧮 Math Problem Solver — Agentic AI

An intelligent **multi-agent system** that parses, solves, verifies, and explains math problems using LangGraph, OpenAI, and RAG-powered retrieval from math textbooks.

> Input a math problem via **text** or **image** → get a step-by-step verified solution with an explanation.

---

## ✨ Features

| Feature | Description |
|---|---|
| 📝 **Multi-Input Support** | Accept math problems as text, images (JPG/PNG), or both |
| 🔍 **OCR Extraction** | Extract math expressions from images using Facebook's [Nougat](https://huggingface.co/facebook/nougat-small) model |
| 🤖 **Multi-Agent Pipeline** | Four specialized agents orchestrated via LangGraph |
| 📚 **RAG Retrieval** | Retrieve relevant formulas & concepts from math textbooks (FAISS + BM25 ensemble) |
| ✅ **Solution Verification** | Confidence-scored verification with automatic re-solve on low scores |
| 👤 **Human-in-the-Loop** | Review & approve parsed problems before solving |
| 🌐 **Web Interface** | Clean FastAPI-powered UI for interactive problem solving |

---

## 🏗️ Architecture

```
                    ┌──────────────┐
                    │  User Input  │
                    │ (text/image) │
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │ Parser Agent │──── OCR (Nougat) if image
                    └──────┬───────┘
                           │
                   ┌───────▼────────┐
                   │  HITL Review   │ ◄── User approves / retries
                   └───────┬────────┘
                           │
                    ┌──────▼───────┐
                    │ Solver Agent │──── Python code execution
                    │              │──── RAG retrieval (FAISS + BM25)
                    └──────┬───────┘
                           │
                   ┌───────▼────────┐
                   │Verifier Agent  │──── Confidence scoring
                   └───────┬────────┘
                           │
                  ┌────────▼─────────┐
                  │Explanation Agent │──── Step-by-step breakdown
                  └──────────────────┘
```

### Agent Details

- **Parser Agent** — Parses raw input into a structured math problem (topic, variables, constraints). Uses Nougat OCR for image inputs.
- **Solver Agent** — Solves the problem using OpenAI with Python code execution and RAG-retrieved formulas/concepts from math textbooks.
- **Verifier Agent** — Cross-checks the solution and assigns a confidence score. Triggers re-solve if confidence is low.
- **Explanation Agent** — Generates a clear, step-by-step explanation of the solution.

---

## 📁 Project Structure

```
Math Solver Agentic Ai/
├── src/
│   ├── agenticai.py       # LangGraph pipeline — builds & runs the multi-agent graph
│   ├── agents.py          # Agent definitions (Parser, Solver, Verifier, Explanation)
│   ├── states.py          # State schema (TypedDict) for the graph
│   ├── ocrmodels.py       # OCR using Facebook Nougat for math image extraction
│   ├── retreival.py       # RAG pipeline — PDF ingestion, FAISS + BM25 retrieval
│   └── main.py            # CLI entry point
├── frontend/
│   ├── main.py            # FastAPI web server with HITL endpoints
│   ├── templates/         # Jinja2 HTML templates
│   └── static/            # CSS & JavaScript assets
├── material/              # Math textbook PDFs (NCERT, IIT JEE)
├── .env                   # API keys (OpenAI)
├── requirements.txt       # Python dependencies
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.10+**
- **OpenAI API Key**
- (Optional) CUDA-enabled GPU for faster OCR inference

### 1. Clone the Repository

```bash
git clone https://github.com/Harsimran726/Math-Problem-Solve-Agentic-Ai.git
cd Math-Problem-Solve-Agentic-Ai
```

### 2. Create a Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the project root:

```env
openai_api_key="your-openai-api-key-here"
```

### 5. Build the FAISS Index (First Time Only)

Run the retrieval script to ingest math textbook PDFs and build the vector store:

```bash
cd src
python retreival.py
```

This creates a `faiss_index/` directory with the prebuilt vector store.

### 6. Run the Application

**Web UI (recommended):**

```bash
cd frontend
uvicorn main:app --reload --port 8000
```

Then open [http://localhost:8000](http://localhost:8000) in your browser.

**CLI mode:**

```bash
cd src
python agenticai.py
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **Orchestration** | [LangGraph](https://github.com/langchain-ai/langgraph) (StateGraph + MemorySaver) |
| **LLM** | OpenAI GPT via [LangChain](https://github.com/langchain-ai/langchain) |
| **OCR** | [Facebook Nougat](https://huggingface.co/facebook/nougat-small) (VisionEncoderDecoder) |
| **Embeddings** | HuggingFace `all-MiniLM-L6-v2` |
| **Vector Store** | FAISS |
| **Retrieval** | Ensemble (FAISS + BM25) |
| **PDF Parsing** | pdfplumber |
| **Web Framework** | FastAPI + Jinja2 |
| **Deep Learning** | PyTorch + Transformers |

---

## 📄 License

This project is for educational purposes.

---

## 🤝 Contributing

Contributions are welcome! Feel free to open an issue or submit a pull request.
