import os
import tempfile
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_groq import ChatGroq

load_dotenv()

st.set_page_config(
    page_title="DocuMind",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .stApp { background: linear-gradient(135deg, #0f0c29, #302b63, #24243e); }
    [data-testid="stSidebar"] { background: rgba(255,255,255,0.05); border-right: 1px solid rgba(255,255,255,0.1); }
    .main-title { font-size: 2.8rem; font-weight: 800; background: linear-gradient(90deg, #a78bfa, #60a5fa, #34d399); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; text-align: center; margin-bottom: 0.2rem; }
    .subtitle { text-align: center; color: rgba(255,255,255,0.5); font-size: 1rem; margin-bottom: 2rem; }
    .chat-user { display: flex; justify-content: flex-end; margin: 0.75rem 0; }
    .chat-user .bubble { background: linear-gradient(135deg, #6d28d9, #4f46e5); color: white; padding: 0.85rem 1.2rem; border-radius: 18px 18px 4px 18px; max-width: 75%; font-size: 0.95rem; box-shadow: 0 4px 15px rgba(109,40,217,0.4); line-height: 1.5; }
    .chat-assistant { display: flex; justify-content: flex-start; margin: 0.75rem 0; }
    .chat-assistant .bubble { background: rgba(255,255,255,0.08); border: 1px solid rgba(255,255,255,0.12); color: rgba(255,255,255,0.92); padding: 0.85rem 1.2rem; border-radius: 18px 18px 18px 4px; max-width: 75%; font-size: 0.95rem; box-shadow: 0 4px 15px rgba(0,0,0,0.3); line-height: 1.6; }
    .avatar { width: 34px; height: 34px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 1rem; flex-shrink: 0; margin: 0 0.5rem; align-self: flex-end; }
    .status-badge { display: inline-block; padding: 0.3rem 0.8rem; border-radius: 999px; font-size: 0.78rem; font-weight: 600; letter-spacing: 0.05em; }
    .badge-ready { background: rgba(52,211,153,0.15); color: #34d399; border: 1px solid rgba(52,211,153,0.3); }
    .stButton > button { background: linear-gradient(135deg, #6d28d9, #4f46e5); color: white; border: none; border-radius: 8px; padding: 0.5rem 1.2rem; font-weight: 600; transition: all 0.2s; }
    .stButton > button:hover { transform: translateY(-1px); box-shadow: 0 6px 20px rgba(109,40,217,0.5); }
    .metric-card { background: rgba(255,255,255,0.06); border: 1px solid rgba(255,255,255,0.1); border-radius: 10px; padding: 0.8rem 1rem; text-align: center; }
    .metric-value { font-size: 1.5rem; font-weight: 700; color: #a78bfa; }
    .metric-label { font-size: 0.75rem; color: rgba(255,255,255,0.5); margin-top: 0.15rem; }
    .chat-container { height: 500px; overflow-y: auto; padding: 1rem; background: rgba(0,0,0,0.2); border-radius: 16px; border: 1px solid rgba(255,255,255,0.08); margin-bottom: 1rem; }
    .chat-container::-webkit-scrollbar { width: 5px; }
    .chat-container::-webkit-scrollbar-thumb { background: rgba(167,139,250,0.4); border-radius: 999px; }
    #MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


def get_api_key():
    key = os.getenv("GROQ_API_KEY", "")
    if not key and "api_key_input" in st.session_state:
        key = st.session_state.api_key_input
    return key


@st.cache_resource(show_spinner=False)
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


@st.cache_resource(show_spinner=False)
def build_vector_store(file_bytes: bytes, filename: str):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    loader = PyPDFLoader(tmp_path)
    documents = loader.load()
    os.unlink(tmp_path)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_documents(documents)
    embeddings = load_embeddings()
    vector_store = FAISS.from_documents(chunks, embeddings)
    return vector_store, len(chunks), len(documents)


def build_qa_chain(vector_store, api_key: str):
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        groq_api_key=api_key,
        temperature=0.2,
    )
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer",
    )
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5},
    )
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        verbose=False,
    )
    return chain


for key, default in {
    "chat_history": [],
    "vector_store": None,
    "qa_chain": None,
    "doc_meta": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

st.markdown('<div class="main-title">🧠 DocuMind</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Intelligent Document Q&A powered by Groq Llama 3.3 + HuggingFace Embeddings</div>', unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    api_key = get_api_key()
    if not api_key:
        st.markdown("**Groq API Key**")
        api_key_input = st.text_input(
            "Enter your API key",
            type="password",
            key="api_key_input",
            placeholder="gsk_...",
            label_visibility="collapsed"
        )
        if api_key_input:
            api_key = api_key_input
        st.caption("Get your key at [console.groq.com](https://console.groq.com/)")
    else:
        st.markdown('<span class="status-badge badge-ready">✓ API Key Loaded</span>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📄 Upload Document")
    uploaded_file = st.file_uploader("Drop a PDF here", type=["pdf"], label_visibility="collapsed")

    if uploaded_file and api_key:
        file_key = f"{uploaded_file.name}_{uploaded_file.size}"
        already_processed = (
            st.session_state.doc_meta is not None
            and st.session_state.doc_meta.get("key") == file_key
        )
        if not already_processed:
            if st.button("🚀 Process Document", use_container_width=True):
                with st.spinner("Loading embeddings model (first time may take 1-2 min)..."):
                    try:
                        vs, n_chunks, n_pages = build_vector_store(
                            uploaded_file.getvalue(),
                            uploaded_file.name,
                        )
                        st.session_state.vector_store = vs
                        st.session_state.qa_chain = build_qa_chain(vs, api_key)
                        st.session_state.chat_history = []
                        st.session_state.doc_meta = {
                            "key": file_key,
                            "name": uploaded_file.name,
                            "pages": n_pages,
                            "chunks": n_chunks,
                        }
                        st.success("Document ready!")
                    except Exception as e:
                        st.error(f"Error: {e}")
        else:
            st.markdown('<span class="status-badge badge-ready">✓ Document Loaded</span>', unsafe_allow_html=True)
    elif uploaded_file and not api_key:
        st.warning("Please enter your Groq API key first.")

    if st.session_state.doc_meta:
        st.markdown("---")
        st.markdown("### 📊 Document Info")
        meta = st.session_state.doc_meta
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{meta["pages"]}</div><div class="metric-label">Pages</div></div>', unsafe_allow_html=True)
        with col2:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{meta["chunks"]}</div><div class="metric-label">Chunks</div></div>', unsafe_allow_html=True)
        st.caption(f"📎 {meta['name']}")

    if st.session_state.chat_history:
        st.markdown("---")
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            if st.session_state.qa_chain:
                st.session_state.qa_chain.memory.clear()
            st.rerun()

    st.markdown("---")
    st.markdown("### 💡 Tips")
    st.caption("• Ask specific questions about the document\n• Follow-up questions use conversation context\n• Answers grounded only in uploaded PDF\n• Embeddings run locally — no API quota!")

if not st.session_state.doc_meta:
    st.markdown("""
    <div style="text-align:center; padding: 4rem 2rem; color: rgba(255,255,255,0.35);">
        <div style="font-size: 4rem; margin-bottom: 1rem;">📂</div>
        <div style="font-size: 1.1rem; font-weight: 600; margin-bottom: 0.5rem; color: rgba(255,255,255,0.5);">No document loaded</div>
        <div style="font-size: 0.9rem;">Upload a PDF in the sidebar and click <strong>Process Document</strong> to begin.</div>
    </div>
    """, unsafe_allow_html=True)
else:
    chat_html = '<div class="chat-container" id="chat-box">'
    if not st.session_state.chat_history:
        chat_html += '<div style="text-align:center; padding: 2rem; color: rgba(255,255,255,0.3); font-size:0.9rem;">Document loaded! Ask your first question below.</div>'
    else:
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                chat_html += f'<div class="chat-user"><div class="bubble">{msg["content"]}</div><div class="avatar">👤</div></div>'
            else:
                content = msg["content"].replace("\n", "<br>")
                chat_html += f'<div class="chat-assistant"><div class="avatar">🧠</div><div class="bubble">{content}</div></div>'
    chat_html += "</div>"
    st.markdown(chat_html, unsafe_allow_html=True)

    if st.session_state.chat_history and "sources" in st.session_state:
        sources = st.session_state.sources
        if sources:
            with st.expander(f"📎 View {len(sources)} source chunk(s) used for last answer"):
                for i, doc in enumerate(sources, 1):
                    page = doc.metadata.get("page", "?")
                    st.markdown(f"**Chunk {i}** — Page {page + 1 if isinstance(page, int) else page}")
                    st.markdown(f'<div style="background:rgba(255,255,255,0.05);border-left:3px solid #a78bfa;padding:0.6rem 1rem;border-radius:0 8px 8px 0;font-size:0.85rem;color:rgba(255,255,255,0.7);">{doc.page_content[:400]}{"…" if len(doc.page_content) > 400 else ""}</div>', unsafe_allow_html=True)

    question = st.chat_input("Ask a question about your document…")
    if question:
        if not st.session_state.qa_chain:
            st.error("Please process a document first.")
        else:
            st.session_state.chat_history.append({"role": "user", "content": question})
            with st.spinner("Thinking…"):
                try:
                    result = st.session_state.qa_chain.invoke({"question": question})
                    answer = result.get("answer", "I could not find an answer in the document.")
                    sources = result.get("source_documents", [])
                    st.session_state.chat_history.append({"role": "assistant", "content": answer})
                    st.session_state.sources = sources
                except Exception as e:
                    st.session_state.chat_history.append({"role": "assistant", "content": f"An error occurred: {str(e)}"})
                    st.session_state.sources = []
            st.rerun()