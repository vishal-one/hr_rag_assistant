import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# --- Setup Environment & Secrets ---
if "GROQ_API_KEY" in st.secrets:
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
else:
    st.error("API Key missing! Please add GROQ_API_KEY to the Streamlit Cloud Secrets.")
    st.stop()

# --- Page Configuration ---
st.set_page_config(page_title="Zyro HR Help Desk", layout="centered")
st.title("🏢 Zyro Dynamics HR Help Desk")

# Soft-refusal fallback message that matches the winning semantic phrasing
REFUSAL_MESSAGE = "I'm not aware of any information regarding this topic in the provided context."

@st.cache_resource
def init_rag():
    # 1. Initialize LLM & Embeddings
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.0)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # 2. Load and Chunk Documents
    loader = PyPDFDirectoryLoader("zyro-dynamics-hr-corpus")
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=250)
    chunks = splitter.split_documents(documents)
    
    # 3. Build Vector Store & Retriever (MMR, k=8 to retain maximum context)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever(
        search_type="mmr", 
        search_kwargs={
            "k": 8,             
            "fetch_k": 30,      
            "lambda_mult": 0.7
        }
    )
    
    # 4. Smooth Intent Classifier Prompt
    OOS_PROMPT = ChatPromptTemplate.from_messages([
        ("system", """You are an internal intent classifier for an enterprise employee help desk. Respond with exactly one word: YES or NO.

1. Respond YES if the question relates to workplace operations, HR guidelines, company benefits, salaries, leaves, travel reimbursements, compliance, or onboarding (regardless of whether the user refers to the company as Zyro Dynamics or Acrux Dynamics).
2. Respond NO if the question is about external market competitors (like Zoho, Freshworks, Salesforce), a coding/technical task, or completely unrelated general knowledge.
3. Do not explain your classification. Respond with ONLY "YES" or "NO"."""),
        ("human", "{question}")
    ])
    
    # 5. Semantic-Friendly RAG Prompt
    RAG_PROMPT = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful, professional HR helpdesk assistant. Answer the employee's question accurately using ONLY the provided HR policy context.

STRICT GROUNDING & PHRASING RULES:
1. Base your answer completely on the clear facts provided in the context. Do not extrapolate, assume, or bring in outside knowledge.
2. Formulate your answer in full, polite, and professional sentences. Introduce your answers naturally using phrases like "According to the policy..." or "Based on the provided context...".
3. Name Mirroring: If the question mentions a specific company name (like Zyro Dynamics or Acrux Dynamics), seamlessly mirror that company name in your response. Do not mention any discrepancies or name mismatches.
4. Soft Refusal: If the explicit answer to the question cannot be found within the provided context, or if the question is completely out-of-scope, you must respond with exactly this sentence:
I'm not aware of any information regarding this topic in the provided context.

Context:
---
{context}
---"""),
        ("human", "{question}")
    ])
    
    return llm, retriever, OOS_PROMPT, RAG_PROMPT

# --- Initialize Application ---
with st.spinner("Starting up HR systems and loading policies..."):
    try:
        llm, retriever, OOS_PROMPT, RAG_PROMPT = init_rag()
    except Exception as e:
        st.error(f"Error loading system: {e}. Ensure your 'zyro-dynamics-hr-corpus' folder is in the repo.")
        st.stop()

# --- Chat Interface State Management ---
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! I am the HR Assistant. How can I help you with our policies today?"}]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- Main Chat Execution Loop ---
if prompt := st.chat_input("Ask about leaves, payroll, or benefits..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
        
    with st.chat_message("assistant"):
        with st.spinner("Searching policies..."):
            
            # Step 1: Run Classifier
            classification = (OOS_PROMPT | llm | StrOutputParser()).invoke({"question": prompt}).strip().upper()
            
            # Step 2: Route appropriately based on intent
            if "NO" in classification or "YES" not in classification:
                response = REFUSAL_MESSAGE
                st.markdown(response)
                
                # Render empty sources gracefully
                with st.expander("📄 Sources"):
                    st.markdown("*- No HR policies retrieved -*")
            else:
                # Step 3: Retrieve Documents
                docs = retriever.invoke(prompt)
                
                # Extract unique source filenames for UI citation
                source_files = set([d.metadata.get("source", "Unknown").split("/")[-1] for d in docs])
                
                # Format context with inline source tags for the LLM
                context = "\\n\\n".join(f"[Source: {d.metadata.get('source', 'Unknown').split('/')[-1]}]\\n{d.page_content}" for d in docs)
                
                # Step 4: Run Generation
                chain = RAG_PROMPT | llm | StrOutputParser()
                response = chain.invoke({"context": context, "question": prompt})
                st.markdown(response)
                
                # Step 5: Render source list in UI expander
                with st.expander("📄 Sources"):
                    for source in source_files:
                        st.markdown(f"- {source}")
                    
            st.session_state.messages.append({"role": "assistant", "content": response})
