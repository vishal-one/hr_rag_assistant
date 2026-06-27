app_code = '''
import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

if "GROQ_API_KEY" in st.secrets:
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
else:
    st.error("API Key missing! Please add GROQ_API_KEY to the Streamlit Cloud Secrets.")
    st.stop()

st.set_page_config(page_title="Zyro HR Help Desk", layout="centered")
st.title("🏢 Zyro Dynamics HR Help Desk")

REFUSAL_MESSAGE = "I can only answer HR-related questions from Zyro Dynamics policy documents."

@st.cache_resource
def init_rag():
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.0)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    loader = PyPDFDirectoryLoader("zyro-dynamics-hr-corpus")
    documents = loader.load()
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=250)
    chunks = splitter.split_documents(documents)
    
    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever(
        search_type="mmr", 
        search_kwargs={"k": 6, "fetch_k": 20, "lambda_mult": 0.7}
    )
    
    OOS_PROMPT = ChatPromptTemplate.from_messages([
        ("system", "You are a strict intent classifier. Respond with exactly one word: YES or NO.\\n\\nIf the question is about company HR policies, travel rules, leaves, benefits, onboarding, code of conduct, or IT compliance, respond YES. If it is an unrelated general question or task request, respond NO."),
        ("human", "{question}")
    ])
    
    RAG_PROMPT = ChatPromptTemplate.from_messages([
        ("system", """You are an expert HR helpdesk assistant. Answer the employee's question accurately using ONLY the provided policy context.

STRICT GROUNDING RULES:
1. Rely ONLY on the clear facts mentioned directly in the context. Do not assume or extrapolate.
2. If the context does not contain the explicit answer to the question, respond with exactly this phrase and nothing else:
I can only answer HR-related questions from Zyro Dynamics policy documents.
3. Answer the question directly and factually. Include specific numbers, dates, timelines, and limits exactly as stated.
4. Do not mention any discrepancy in company names. Just answer the question.

Context:
{context}"""),
        ("human", "{question}")
    ])
    
    return llm, retriever, OOS_PROMPT, RAG_PROMPT

with st.spinner("Starting up HR systems and loading policies..."):
    try:
        llm, retriever, OOS_PROMPT, RAG_PROMPT = init_rag()
    except Exception as e:
        st.error(f"Error loading system: {e}.")
        st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! I am the HR Assistant. How can I help you with our policies today?"}]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask about leaves, payroll, or benefits..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
        
    with st.chat_message("assistant"):
        with st.spinner("Searching policies..."):
            classification = (OOS_PROMPT | llm | StrOutputParser()).invoke({"question": prompt}).strip().upper()
            
            if "NO" in classification or "YES" not in classification:
                response = REFUSAL_MESSAGE
                st.markdown(response)
            else:
                docs = retriever.invoke(prompt)
                context = "\\n\\n".join(d.page_content for d in docs)
                
                chain = RAG_PROMPT | llm | StrOutputParser()
                response = chain.invoke({"context": context, "question": prompt})
                st.markdown(response)
                
                with st.expander("View Retrieved Context"):
                    st.text(context)
                    
            st.session_state.messages.append({"role": "assistant", "content": response})
'''

with open("app.py", "w") as f:
    f.write(app_code.strip())
print("app.py updated successfully.")
