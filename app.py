
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from ddgs import DDGS
def search_web(query):
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5))

        search_context = ""

        for result in results:
            search_context += (
                f"\nTitle: {result.get('title', '')}\n"
                f"URL: {result.get('href', '')}\n"
                f"Description: {result.get('body', '')}\n"
            )

        return search_context

    except Exception as e:
        return f"Search error: {e}"

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Sai AI Chatbot",
    page_icon="🤖",
    layout="centered"
)

# ---------------- CUSTOM UI ----------------
st.markdown("""
<style>
.stApp {
    background-color: #0e1117;
}
[data-testid="stChatMessage"] {
    border-radius: 14px;
    padding: 12px;
}
.main-title {
    text-align: center;
    font-size: 34px;
    font-weight: bold;
    color: #8b5cf6;
}
.subtitle {
    text-align: center;
    color: #9ca3af;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🤖 Sai AI</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Your personal AI assistant</div>',
    unsafe_allow_html=True
)

# ---------------- MODEL ----------------
@st.cache_resource
def load_model():
    model_name = "HuggingFaceTB/SmolLM2-135M-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    model.eval()
    return tokenizer, model

with st.spinner("Loading AI model..."):
    tokenizer, model = load_model()

# ---------------- SYSTEM PROMPT ----------------
SYSTEM_PROMPT = """
You are Sai AI, a friendly and helpful AI assistant.
Answer clearly and simply.
Use examples when useful.
Remember the context of the conversation while responding.
"""

# ---------------- SESSION MEMORY ----------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.title("⚙️ Settings")

    st.write("Model: SmolLM2-135M-Instruct")

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]
        st.rerun()

    st.caption("Sai AI • Built with Streamlit")

# ---------------- DISPLAY CHAT ----------------
for message in st.session_state.messages:
    if message["role"] == "system":
        continue

    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ---------------- USER INPUT ----------------
web_search_enabled = st.sidebar.checkbox(
    "🌐 Enable Web Search",
    value=False
)
user_input = st.chat_input("Message Sai AI...")

if user_input:
    st.session_state.messages.append(
        {"role": "user", "content": user_input}
    )

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Sai AI is thinking..."):

            inputs = tokenizer.apply_chat_template(
                st.session_state.messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True
            )

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=180,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    repetition_penalty=1.1
                )

            input_length = inputs["input_ids"].shape[1]
            new_tokens = outputs[0][input_length:]

            response = tokenizer.decode(
                new_tokens,
                skip_special_tokens=True
            ).strip()

            if not response:
                response = "I couldn't generate a response. Please try again."

            st.markdown(response)

    st.session_state.messages.append(
        {"role": "assistant", "content": response}
    )
