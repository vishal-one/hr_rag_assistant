import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# --- ADD THESE 3 LINES TO FIX THE AUTH ERROR ---
if "GROQ_API_KEY" in st.secrets:
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
# -----------------------------------------------

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
        ("system", "You are an HR classifier for Zyro Dynamics. Respond with ONLY one word: YES (if HR-related) or NO (if not HR-related)."),
        ("human", "{question}")
    ])
    
    # 5. RAG Prompt

    # 5. RAG Prompt
    RAG_PROMPT = ChatPromptTemplate.from_messages([
        ("system", "You are a strict HR assistant for Zyro Dynamics. Answer the question using ONLY the provided context.\\n\\nCRITICAL RULE: If the exact answer is not explicitly stated in the context, do NOT use outside knowledge and do NOT explain yourself. You MUST respond with exactly this phrase and nothing else: I can only answer HR-related questions from Zyro Dynamics policy documents.\\n\\nContext:\\n{context}"),
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
