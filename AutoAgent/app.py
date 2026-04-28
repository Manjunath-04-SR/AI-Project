import os
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.prebuilt import create_react_agent

load_dotenv()

st.set_page_config(
    page_title="AutoAgent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .stApp { background: linear-gradient(135deg, #0a0a1a, #0d1b2a, #1a0a2e); }
    [data-testid="stSidebar"] { background: rgba(255,255,255,0.04); border-right: 1px solid rgba(255,255,255,0.08); }
    .main-title { font-size: 2.8rem; font-weight: 800; background: linear-gradient(90deg, #f59e0b, #ef4444, #8b5cf6); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; text-align: center; margin-bottom: 0.2rem; }
    .subtitle { text-align: center; color: rgba(255,255,255,0.45); font-size: 1rem; margin-bottom: 2rem; }
    .chat-user { display: flex; justify-content: flex-end; margin: 0.75rem 0; }
    .chat-user .bubble { background: linear-gradient(135deg, #b45309, #92400e); color: white; padding: 0.85rem 1.2rem; border-radius: 18px 18px 4px 18px; max-width: 75%; font-size: 0.95rem; line-height: 1.5; }
    .chat-assistant { display: flex; justify-content: flex-start; margin: 0.75rem 0; }
    .chat-assistant .bubble { background: rgba(255,255,255,0.07); border: 1px solid rgba(255,255,255,0.1); color: rgba(255,255,255,0.9); padding: 0.85rem 1.2rem; border-radius: 18px 18px 18px 4px; max-width: 80%; font-size: 0.95rem; line-height: 1.6; }
    .tool-badge { display: inline-block; background: rgba(245,158,11,0.15); color: #f59e0b; border: 1px solid rgba(245,158,11,0.3); padding: 0.2rem 0.6rem; border-radius: 999px; font-size: 0.75rem; font-weight: 600; margin: 0.2rem; }
    .step-card { background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.08); border-radius: 10px; padding: 0.8rem 1rem; margin: 0.4rem 0; }
    .step-label { font-size: 0.75rem; font-weight: 700; color: #f59e0b; text-transform: uppercase; letter-spacing: 0.08em; }
    .step-content { font-size: 0.88rem; color: rgba(255,255,255,0.75); margin-top: 0.25rem; line-height: 1.5; }
    .status-badge { display: inline-block; padding: 0.3rem 0.8rem; border-radius: 999px; font-size: 0.78rem; font-weight: 600; }
    .badge-ready { background: rgba(52,211,153,0.15); color: #34d399; border: 1px solid rgba(52,211,153,0.3); }
    .stButton > button { background: linear-gradient(135deg, #b45309, #92400e); color: white; border: none; border-radius: 8px; font-weight: 600; }
    .chat-container { height: 520px; overflow-y: auto; padding: 1rem; background: rgba(0,0,0,0.25); border-radius: 16px; border: 1px solid rgba(255,255,255,0.07); margin-bottom: 1rem; }
    .chat-container::-webkit-scrollbar { width: 4px; }
    .chat-container::-webkit-scrollbar-thumb { background: rgba(245,158,11,0.4); border-radius: 999px; }
    #MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


def get_groq_key():
    key = os.getenv("GROQ_API_KEY", "")
    if not key and "groq_key_input" in st.session_state:
        key = st.session_state.groq_key_input
    return key


def get_tavily_key():
    key = os.getenv("TAVILY_API_KEY", "")
    if not key and "tavily_key_input" in st.session_state:
        key = st.session_state.tavily_key_input
    return key


@st.cache_resource(show_spinner=False)
def build_agent(groq_key: str, tavily_key: str):
    os.environ["TAVILY_API_KEY"] = tavily_key
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        groq_api_key=groq_key,
        temperature=0.3,
    )
    tools = [TavilySearchResults(max_results=4)]
    system_prompt = """You are AutoAgent — a powerful AI assistant with access to real-time web search.

Your capabilities:
- Answer questions using real-time web search
- Research and summarize topics
- Find latest news and information
- Analyze and reason over search results
- Perform multi-step research tasks

When you use the search tool:
- Search for specific, targeted queries
- Synthesize information from multiple sources
- Always provide clear, well-structured answers
- Cite what you found in your response

Be helpful, thorough, and accurate."""

    agent = create_react_agent(
        model=llm,
        tools=tools,
        prompt=system_prompt,
    )
    return agent


# ── Session state ─────────────────────────────────────────────────────────────
for key, default in {
    "chat_history": [],
    "agent_steps": {},
    "agent": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">🤖 AutoAgent</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Autonomous AI Agent with Real-Time Web Search — Powered by Groq Llama 3.3 + LangGraph</div>', unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Configuration")

    groq_key = get_groq_key()
    tavily_key = get_tavily_key()

    if not groq_key:
        st.markdown("**Groq API Key**")
        groq_input = st.text_input("Groq Key", type="password", key="groq_key_input",
                                    placeholder="gsk_...", label_visibility="collapsed")
        if groq_input:
            groq_key = groq_input
    else:
        st.markdown('<span class="status-badge badge-ready">✓ Groq Key Loaded</span>', unsafe_allow_html=True)

    st.markdown("")

    if not tavily_key:
        st.markdown("**Tavily API Key**")
        tavily_input = st.text_input("Tavily Key", type="password", key="tavily_key_input",
                                      placeholder="tvly-...", label_visibility="collapsed")
        if tavily_input:
            tavily_key = tavily_input
        st.caption("Get free key at [app.tavily.com](https://app.tavily.com/)")
    else:
        st.markdown('<span class="status-badge badge-ready">✓ Tavily Key Loaded</span>', unsafe_allow_html=True)

    if groq_key and tavily_key:
        st.markdown("---")
        if st.session_state.agent is None:
            with st.spinner("Initializing agent..."):
                try:
                    st.session_state.agent = build_agent(groq_key, tavily_key)
                    st.success("Agent ready!")
                except Exception as e:
                    st.error(f"Error: {e}")

    st.markdown("---")
    st.markdown("### 🛠️ Available Tools")
    st.markdown('<span class="tool-badge">🔍 Web Search</span>', unsafe_allow_html=True)
    st.markdown('<span class="tool-badge">🧠 Reasoning</span>', unsafe_allow_html=True)
    st.markdown('<span class="tool-badge">📊 Analysis</span>', unsafe_allow_html=True)
    st.markdown('<span class="tool-badge">📝 Summarization</span>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 💡 Try Asking")
    st.caption("• What are the latest AI developments in 2026?")
    st.caption("• Compare Python vs JavaScript for AI development")
    st.caption("• What is LangGraph and how does it work?")
    st.caption("• Latest news about Cognizant")
    st.caption("• Explain RAG pipelines and their use cases")

    if st.session_state.chat_history:
        st.markdown("---")
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.session_state.agent_steps = {}
            st.rerun()

    st.markdown("---")
    st.markdown("### 🤖 Model Info")
    st.caption("LLM: Groq Llama 3.3 70B")
    st.caption("Search: Tavily Real-Time Web")
    st.caption("Framework: LangGraph ReAct Agent")

# ── Main Chat Area ────────────────────────────────────────────────────────────
if not groq_key or not tavily_key:
    st.markdown("""
    <div style="text-align:center; padding:4rem 2rem; color:rgba(255,255,255,0.3);">
        <div style="font-size:4rem; margin-bottom:1rem;">🤖</div>
        <div style="font-size:1.1rem; font-weight:600; color:rgba(255,255,255,0.5); margin-bottom:0.5rem;">Agent Not Configured</div>
        <div style="font-size:0.9rem;">Enter your Groq and Tavily API keys in the sidebar to activate the agent.</div>
    </div>
    """, unsafe_allow_html=True)
else:
    # Chat display
    chat_html = '<div class="chat-container" id="chat-box">'
    if not st.session_state.chat_history:
        chat_html += """
        <div style="text-align:center; padding:2rem; color:rgba(255,255,255,0.3); font-size:0.9rem;">
            Agent is ready! Ask me anything — I can search the web in real time.
        </div>"""
    else:
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                chat_html += f'<div class="chat-user"><div class="bubble">{msg["content"]}</div><div style="margin-left:0.5rem;font-size:1.2rem;align-self:flex-end;">👤</div></div>'
            else:
                content = msg["content"].replace("\n", "<br>")
                chat_html += f'<div class="chat-assistant"><div style="margin-right:0.5rem;font-size:1.2rem;align-self:flex-end;">🤖</div><div class="bubble">{content}</div></div>'
    chat_html += "</div>"
    st.markdown(chat_html, unsafe_allow_html=True)

    # Show agent steps for last message
    if st.session_state.chat_history:
        last_idx = len(st.session_state.chat_history) - 1
        steps = st.session_state.agent_steps.get(last_idx, [])
        if steps:
            with st.expander(f"🔍 View Agent Reasoning Steps ({len(steps)} steps)"):
                for i, step in enumerate(steps, 1):
                    st.markdown(f"""
                    <div class="step-card">
                        <div class="step-label">Step {i} — {step['type']}</div>
                        <div class="step-content">{step['content']}</div>
                    </div>
                    """, unsafe_allow_html=True)

    # Question input
    question = st.chat_input("Ask AutoAgent anything — it will search the web if needed...")

    if question:
        if st.session_state.agent is None:
            st.error("Agent not initialized. Please check your API keys.")
        else:
            st.session_state.chat_history.append({"role": "user", "content": question})
            msg_idx = len(st.session_state.chat_history)

            with st.spinner("🤖 Agent is thinking and searching..."):
                try:
                    steps = []
                    messages = [HumanMessage(content=question)]
                    result = st.session_state.agent.invoke({"messages": messages})

                    # Extract steps from messages
                    for msg in result.get("messages", []):
                        if hasattr(msg, 'tool_calls') and msg.tool_calls:
                            for tc in msg.tool_calls:
                                steps.append({
                                    "type": f"🔍 Tool Call: {tc['name']}",
                                    "content": f"Searching for: {tc['args'].get('query', str(tc['args']))}"
                                })
                        elif hasattr(msg, 'name') and msg.name:
                            content = str(msg.content)[:500]
                            steps.append({
                                "type": f"📥 Tool Result: {msg.name}",
                                "content": content + ("..." if len(str(msg.content)) > 500 else "")
                            })

                    # Get final answer
                    final_message = result["messages"][-1]
                    answer = final_message.content

                    st.session_state.chat_history.append({"role": "assistant", "content": answer})
                    st.session_state.agent_steps[msg_idx] = steps

                except Exception as e:
                    error_msg = f"An error occurred: {str(e)}"
                    st.session_state.chat_history.append({"role": "assistant", "content": error_msg})
                    st.session_state.agent_steps[msg_idx] = []

            st.rerun()