import streamlit as st
import os

st.set_page_config(page_title="Zyro Dynamics HR Help Desk", page_icon="🏢", layout="wide")

with st.sidebar:
    st.title("🏢 Zyro Dynamics HR Help Desk")
    st.markdown("---")
    top_k = st.slider("Chunks to retrieve", 3, 10, 5)
    st.markdown("---")
    st.markdown("**📂 HR Policies:**")
    for p in ["Company Profile","Employee Handbook","Leave Policy",
              "WFH Policy","Code of Conduct","Performance Review",
              "Compensation & Benefits","IT & Data Security",
              "POSH Policy","Onboarding & Separation","Travel & Expense"]:
        st.markdown(f"• {p}")
    if st.button("🗑️ Clear chat"):
        st.session_state.messages = []
        st.rerun()

@st.cache_resource(show_spinner="Loading HR policies...")
def load_pipeline():
    os.environ["GROQ_API_KEY"]         = st.secrets["GROQ_API_KEY"]
    os.environ["LANGCHAIN_API_KEY"]    = st.secrets["LANGCHAIN_API_KEY"]
    os.environ["LANGCHAIN_PROJECT"]    = "zyro-rag-challenge"
    os.environ["LANGCHAIN_TRACING_V2"] = "true"

    from langchain_community.document_loaders import PyPDFDirectoryLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS
    from langchain_groq import ChatGroq

    docs   = PyPDFDirectoryLoader("./hr_policies/").load()
    chunks = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=80).split_documents(docs)
    emb    = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vs     = FAISS.from_documents(chunks, emb)
    llm    = ChatGroq(model="llama-3.1-8b-instant", temperature=0, max_tokens=512)
    return vs, llm

vs, llm = load_pipeline()

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

OOS_PROMPT = ChatPromptTemplate.from_template(
    "Is this question about HR policies, leave, payroll, benefits, conduct, "
    "WFH, performance, onboarding, travel, or IT security at Zyro Dynamics? "
    "Reply YES or NO only.\n\nQuestion: {question}\nAnswer:"
)
RAG_PROMPT = ChatPromptTemplate.from_template(
    "You are Zyro Dynamics HR Help Desk assistant.\n"
    "Answer ONLY from the context below. Cite the source policy.\n\n"
    "Context: {context}\n\nQuestion: {question}\nAnswer:"
)
REFUSAL = (
    "I can only answer Zyro Dynamics HR policy questions "
    "(leave, payroll, benefits, WFH, conduct, etc.). "
    "For other queries, please contact the relevant team."
)

def ask(question, k=5):
    verdict = (OOS_PROMPT | llm | StrOutputParser()).invoke({"question": question}).strip().upper()
    if not verdict.startswith("YES"):
        return REFUSAL, [], True
    ret   = vs.as_retriever(search_type="mmr", search_kwargs={"k": k, "fetch_k": 20})
    docs  = ret.invoke(question)
    chain = (
        {"context": ret | (lambda d: "\n\n".join(
            f"[{os.path.basename(x.metadata.get('source','?'))}]\n{x.page_content}" for x in d
        )), "question": RunnablePassthrough()}
        | RAG_PROMPT | llm | StrOutputParser()
    )
    answer  = chain.invoke(question)
    sources = list({os.path.basename(d.metadata.get("source","")) for d in docs})
    return answer, sources, False

st.title("🏢 Zyro Dynamics HR Help Desk")
st.caption("Ask about leave, payroll, WFH, benefits, conduct, performance, and more.")

if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({
        "role": "assistant",
        "content": "Hi! I am your Zyro Dynamics HR assistant. Ask me about leave, payroll, WFH, benefits, or any HR policy.",
        "sources": []
    })

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            st.caption("📄 Sources: " + " · ".join(msg["sources"]))

if user_q := st.chat_input("Ask an HR question..."):
    st.session_state.messages.append({"role": "user", "content": user_q, "sources": []})
    with st.chat_message("user"):
        st.markdown(user_q)
    with st.chat_message("assistant"):
        with st.spinner("Searching HR policies..."):
            answer, sources, blocked = ask(user_q, top_k)
        st.markdown(answer)
        if sources:
            st.caption("📄 Sources: " + " · ".join(sources))
        if blocked:
            st.caption("⚠️ Out-of-scope question.")
    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})