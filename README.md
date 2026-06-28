# 🏢 Zyro Dynamics HR Help Desk (RAG Pipeline)

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![LangChain](https://img.shields.io/badge/LangChain-Enabled-green)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B)
![LLM](https://img.shields.io/badge/LLM-Llama_3.1-orange)

An intelligent, production-ready Retrieval-Augmented Generation (RAG) chatbot built to automate complex HR policy inquiries for Zyro Dynamics. This project was developed as part of a competitive Kaggle challenge, achieving top-tier accuracy against a strict automated semantic similarity grader.

## 🚀 Overview

This project implements a highly optimized RAG pipeline that reads, chunks, and retrieves information from 11 internal HR policy PDFs to answer employee questions. 

Unlike basic RAG setups, this pipeline is engineered to survive adversarial data traps (like intentionally swapped company names) and strict phrasing evaluations. It features a dual-layer prompt architecture, advanced vector retrieval, and a fully interactive UI.

## ✨ Key Features

* **Adversarial Data Defense:** Implements a strict "Intent Classifier" to dynamically bypass data poisoning (interchangeable use of "Zyro Dynamics" and "Acrux Dynamics") while strictly blocking out-of-scope queries about external competitors.
* **Semantic-Optimized Generation:** Uses tailored conversational framing and "soft refusal" fallback logic to maximize semantic similarity scores against hidden ground truths.
* **Maximal Marginal Relevance (MMR) Retrieval:** Replaces standard cosine similarity with MMR (`k=8`, `fetch_k=30`) to guarantee a diverse, multi-document context window for highly complex questions.
* **Live Citation UI:** Dynamically extracts document metadata to render real-time source file citations in the Streamlit frontend.
* **Full Observability:** Integrated with LangSmith for token tracking, latency monitoring, and LLM output debugging.

## 🛠️ Tech Stack

* **Orchestration:** [LangChain](https://python.langchain.com/)
* **LLM:** [Groq](https://groq.com/) (Llama-3.1-8b-instant) for ultra-low latency inference
* **Embeddings:** HuggingFace (`sentence-transformers/all-MiniLM-L6-v2`)
* **Vector Database:** FAISS (Facebook AI Similarity Search)
* **Frontend:** [Streamlit](https://streamlit.io/)
* **Tracing/Monitoring:** LangSmith

## 📂 Repository Structure

```text
├── zyro-dynamics-hr-corpus/    # Directory containing the 11 HR Policy PDFs
├── app.py                      # Main Streamlit application and RAG backend
├── __notebook_source__.ipynb   # Original Kaggle notebook for pipeline testing and submission generation
├── requirements.txt            # Project dependencies
└── README.md


⚙️ Quick Start
1. Clone the repository
Bash
git clone [https://github.com/yourusername/zyro-hr-rag.git](https://github.com/yourusername/zyro-hr-rag.git)
cd zyro-hr-rag
2. Install dependencies
Bash
pip install -r requirements.txt
3. Set up environment variables
Create a .streamlit/secrets.toml file in the root directory and add your API keys:

Ini, TOML
GROQ_API_KEY = "your_groq_api_key_here"
(Optional: If running the Jupyter Notebook locally, add your LANGCHAIN_API_KEY to track traces in LangSmith).

4. Run the application
Bash
streamlit run app.py
🧠 The "Data Poisoning" Challenge
A major hurdle in this project was intentional dataset obfuscation. The source PDFs randomly swapped the company name between "Zyro Dynamics" and "Acrux Dynamics." Standard RAG architectures either hallucinated policies or refused valid questions due to entity mismatches.

The Solution: Instead of hardcoding keyword blocks, I engineered an overarching Intent Classifier prompt that conceptually merged the two entities, allowing the downstream RAG Generator to safely pull cross-company context without triggering out-of-scope guardrails
