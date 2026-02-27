# 🤖 Vernacular Employee Training Bot

An AI-powered RAG (Retrieval-Augmented Generation) chatbot that lets employees query company policy documents in their native language — with both a written answer and an automatic voice response.

## ✨ Features

- 📄 **PDF Knowledge Base** — Upload any company policy PDF; the bot indexes it instantly using FAISS
- 🧠 **Dual LLM** — Meta Llama 3-8B (primary) with Qwen 2.5-7B as automatic fallback
- 💡 **Two-Part Answers** — Every response includes a document-grounded answer + a plain-language explanation
- 🔊 **Auto Voice** — Offline text-to-speech (macOS `say` + `ffmpeg`) plays automatically after each response
- 🌐 **Multilingual** — English, Hindi, Telugu
- ⚡ **Semantic Cache** — Repeated questions answered instantly from cache
- 🎨 **Sci-Fi Dark UI** — Space-themed dashboard with animated particles

## 🏗️ Architecture

```
PDF Upload → Text Chunking → FAISS Embedding → RAG Query → LLM Answer → TTS Voice
```

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- macOS (for offline TTS via `say` command)
- `ffmpeg` installed (`brew install ffmpeg`)
- Hugging Face account with API key

### Installation

```bash
git clone https://github.com/AthotiVenkatLakshman/AI-Based-Face-Emotion-Detection-for-Interview-Readiness.git
cd AI-Based-Face-Emotion-Detection-for-Interview-Readiness/hackathon

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Configure API Key

Create a `.env` file in the project root:
```
HF_API_KEY=your_huggingface_api_key_here
```

Get a free key at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

### Run

```bash
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

## 📁 Project Structure

```
hackathon/
├── app.py              # Streamlit UI (sci-fi dark dashboard)
├── rag_pipeline.py     # TrainingBot: PDF ingestion, FAISS, LLM
├── utils.py            # TTS (offline), chat history, language mapping
├── requirements.txt    # Python dependencies
└── .env                # API keys (NOT committed)
```

## 🤖 Models Used

| Model | Role |
|---|---|
| `meta-llama/Meta-Llama-3-8B-Instruct` | Primary LLM |
| `Qwen/Qwen2.5-7B-Instruct` | Fallback LLM |
| `sentence-transformers/all-MiniLM-L6-v2` | Text embeddings |

## 🛠️ Tech Stack

`Streamlit` · `LangChain` · `FAISS` · `HuggingFace Inference API` · `pypdf` · `ffmpeg`

## 📝 USP

> **AI-powered corporate training assistant that answers policy questions in the employee's native language — with a document-grounded answer, explanation, and automatic offline voice response.**
