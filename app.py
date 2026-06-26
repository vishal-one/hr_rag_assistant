import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# --- THIS IS THE CRITICAL CHANGE ---
# Pull the key from Streamlit's secure vault and give it to LangChain
if "GROQ_API_KEY" in st.secrets:
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
else:
    st.error("API Key missing! Please add GROQ_API_KEY to the Streamlit Cloud Secrets.")
    st.stop()
# -----------------------------------

st.set_page_config(page_title="Zyro HR Help Desk", layout="centered")
st.title("🏢 Zyro Dynamics HR Help Desk")

# The exact Kaggle refusal string required for a perfect score
REFUSAL_MESSAGE = "I can only answer HR-related questions from Zyro Dynamics policy documents."

@st.cache_resource
def init_rag():
    # 1. Initialize LLM (Ensure GROQ_API_KEY is in Streamlit secrets)
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.0)
    
    # 2. Initialize Embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # 3. Load and build FAISS vector store on startup
    loader = PyPDFDirectoryLoader("zyro-dynamics-hr-corpus")
    documents = loader.load()
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=250)
    chunks = splitter.split_documents(documents)
    
    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 8})
    
   # 4. Out-of-Scope Prompt
    OOS_PROMPT = ChatPromptTemplate.from_messages([
        ("system", "You are a strict classifier. Your ONLY approved company is 'Zyro Dynamics'. If the user's question mentions, implies, or asks about ANY other company name or organization, you must respond with EXACTLY the word NO. If the question is not about HR, respond with NO. Otherwise, respond with YES."),
        ("human", "{question}")
    ])
    
    # 5. RAG Prompt
    RAG_PROMPT = ChatPromptTemplate.from_messages([
        ("system", "You are a strict HR assistant for Zyro Dynamics. Answer the question using ONLY the provided context.

CRITICAL RULE: If the user asks about policies for any company other than Zyro Dynamics, OR if the exact answer is not explicitly stated in the context, do NOT explain yourself. You MUST respond with exactly this phrase and nothing else: I can only answer HR-related questions from Zyro Dynamics policy documents.

Context:
{context}"),
        ("human", "{question}")
    ])
    return llm, retriever, OOS_PROMPT, RAG_PROMPT

# Initialize the system (this only runs once thanks to @st.cache_resource)
with st.spinner("Starting up HR systems and loading policies..."):
    try:
        llm, retriever, OOS_PROMPT, RAG_PROMPT = init_rag()
    except Exception as e:
        st.error(f"Error loading system: {e}. Ensure your 'zyro-dynamics-hr-corpus' folder is in the repo.")
        st.stop()

# Chat UI State
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Handle Chat Input
if prompt := st.chat_input("Ask about leaves, payroll, or benefits..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
        
    with st.chat_message("assistant"):
        with st.spinner("Searching policies..."):
            # Step 1: Guardrail Check
            classification = (OOS_PROMPT | llm | StrOutputParser()).invoke({"question": prompt}).strip().upper()
            
            if "NO" in classification:
                response = REFUSAL_MESSAGE
            else:
                # Step 2: RAG Pipeline Execution
                docs = retriever.invoke(prompt)
                context = "\n\n".join(d.page_content for d in docs)
                
                chain = RAG_PROMPT | llm | StrOutputParser()
                response = chain.invoke({"context": context, "question": prompt})
            
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})