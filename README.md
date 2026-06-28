# 🏢 Zyro Dynamics HR Help Desk — Kaggle RAG Challenge

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![LangChain](https://img.shields.io/badge/LangChain-Enabled-green)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B)
![LLM](https://img.shields.io/badge/LLM-Llama_3.1-orange)

An intelligent, production-ready Retrieval-Augmented Generation (RAG) chatbot built to automate complex HR policy inquiries for Zyro Dynamics. This repository contains my solution for the **NIAT Masterclass - RAG Challenge**, where this pipeline secured a top-tier **89.11% accuracy score** on the competitive leaderboard.

## 🏆 Kaggle Competition Context

The hackathon challenged participants to build a highly robust enterprise virtual assistant capable of querying 11 complex internal corporate policy PDFs. Submissions were evaluated by an automated backend grading script using strict **semantic similarity metrics** against a hidden ground-truth answer key. 

To prevent basic keyword-matching shortcuts, the evaluation set contained adversarial "traps," out-of-scope general knowledge queries, and intentional data contradictions designed to break fragile pipelines.

## 🚀 Overview

This project implements an end-to-end optimized RAG pipeline capable of handling adversarial data setups while keeping generation tightly grounded. Unlike standard "plug-and-play" RAG tutorials, this architecture features a dual-layer prompt defense framework, diverse vector space retrieval, and an interactive frontend wrapper.

## ✨ Key Features

* **Adversarial Data Defense:** Implements a strict "Intent Classifier" guardrail to dynamically bypass data poisoning (interchangeable use of "Zyro Dynamics" and "Acrux Dynamics") while cleanly blocking out-of-scope queries about external tech competitors.
* **Semantic-Optimized Phrasing:** Utilizes custom-tailored corporate framing guidelines and precise "soft refusal" fallback logic to match the exact tone and structure expected by the Kaggle auto-grader.
* **Maximal Marginal Relevance (MMR) Retrieval:** Replaces standard cosine similarity with MMR (`k=8`, `fetch_k=30`) to balance query relevance with chunk diversity, preventing the LLM from missing crucial context scattered across different policy updates.
* **Live Citation UI:** Dynamically parses chunk metadata to render real-time source file citations directly in the Streamlit frontend.
* **Production-Grade Observability:** Fully integrated with LangSmith for deep execution tracing, latency analysis, and token tracking.

## 🛠️ Tech Stack

* **Orchestration:** [LangChain](https://python.langchain.com/) (LCEL)
* **LLM:** [Groq](https://groq.com/) (Llama-3.1-8b-instant) for lightning-fast inference
* **Embeddings:** HuggingFace (`sentence-transformers/all-MiniLM-L6-v2`)
* **Vector Database:** FAISS (Facebook AI Similarity Search)
* **Frontend:** [Streamlit](https://streamlit.io/)
* **Monitoring:** LangSmith

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
A major hurdle in this project was intentional dataset obfuscation.
The source PDFs randomly swapped the company name between "Zyro Dynamics" and "Acrux Dynamics." 
Standard RAG architectures either hallucinated policies or refused valid questions due to entity mismatches.

The Solution: Instead of hardcoding keyword blocks.
I engineered an overarching Intent Classifier prompt that conceptually merged the two entities,
Allowing the downstream RAG Generator to safely pull cross-company context without triggering out-of-scope guardrails
