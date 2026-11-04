
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from ddgs import DDGS
from pypdf import PdfReader

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Sai AI Chatbot",
    page_icon="🤖",
    layout="centered"
)

# ---------------- WEB SEARCH ----------------
def search_web(query):
    try:
        with DDGS() as ddgs:
            results = list(
                ddgs.news(
                    query,
                    region="us-en",
                    max_results=5
                )
            )

        return results

    except Exception as e:
        print("Search error:", e)
        return []


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

st.markdown(
    '<div class="main-title">🤖 Sai AI</div>',
    unsafe_allow_html=True
)

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
You are Sai AI, a friendly, intelligent, and natural AI assistant.

Your personality:
- Talk like a real helpful friend.
- Be warm, friendly, and conversational.
- Do not sound like a textbook or robot.
- Understand the user's intention before answering.
- Give direct answers first.
- Explain complex topics in simple language.
- Use examples and analogies when useful.
- Keep answers concise for simple questions.
- Give detailed explanations when the user asks.
- Ask a question only when necessary.
- Remember the conversation context available in the chat history.
- Never invent facts.
- If you don't know something, say so honestly.

For casual conversations:
Talk naturally, like a close and helpful friend.

For technical questions:
Explain step by step with simple examples.

For coding questions:
Explain the code, the logic, and why it works.

Your goal is to make the user feel that they are
having a natural conversation with a helpful AI assistant.
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

    st.divider()
    st.subheader("📄 Chat with PDF")

    uploaded_pdf = st.file_uploader(
        "Upload your PDF",
        type=["pdf"]
    )

    pdf_context = ""

    if uploaded_pdf is not None:
        reader = PdfReader(uploaded_pdf)

        for page in reader.pages:
            text = page.extract_text()

            if text:
                pdf_context += text + "\n"

        st.success("PDF uploaded successfully!")

    st.caption("Sai AI • Built with Streamlit")

# ---------------- DISPLAY CHAT ----------------
for message in st.session_state.messages:
    if message["role"] == "system":
        continue

    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ---------------- USER INPUT ----------------
mode = st.sidebar.radio(
    "Choose Mode",
    ["🤖 AI Chat", "🌐 AI News Search", "📄 Chat with PDF"]
)

web_search_enabled = mode == "🌐 AI News Search"

user_input = st.chat_input("Message Sai AI...")

if user_input:

    original_input = user_input

    # PDF MODE
    if pdf_context and mode == "📄 Chat with PDF":
        user_input = f"""
Answer the question using the uploaded PDF content.

PDF Content:
{pdf_context[:4000]}

Question:
{original_input}

If the answer is not in the PDF, say:
"I couldn't find this information in the PDF."
"""

    # Save user message
    st.session_state.messages.append(
        {"role": "user", "content": original_input}
    )

    # Display user message
    with st.chat_message("user"):
        st.markdown(original_input)

    # ---------------- WEB SEARCH MODE ----------------
    if web_search_enabled:

        with st.chat_message("assistant"):
            with st.spinner("Searching the latest news..."):

                results = search_web(user_input)

                if results:

                    st.markdown("### 🌐 Latest News Results")

                    news_text = ""

                    for i, result in enumerate(results, 1):
                        title = result.get("title", "No title")
                        date = result.get("date", "Date unavailable")
                        source = result.get("source", "Source unavailable")
                        url = result.get("url", "")
                        body = result.get("body", "No description available")

                        st.markdown(f"### {i}. {title}")
                        st.caption(f"📅 {date} | 📰 {source}")
                        st.write(body)

                        if url:
                            st.markdown(f"[Read full article]({url})")

                        st.divider()

                        news_text += (
                            f"{i}. {title}\n"
                            f"Date: {date}\n"
                            f"Source: {source}\n"
                            f"URL: {url}\n"
                            f"Description: {body}\n\n"
                        )

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": news_text
                        }
                    )

                else:
                    st.warning(
                        "No news results found. Please try another search."
                    )

    # ---------------- NORMAL AI CHAT MODE ----------------
    else:

        with st.chat_message("assistant"):
            with st.spinner("Sai AI is thinking..."):

                model_messages = st.session_state.messages.copy()

                # PDF context for model
                if pdf_context:
                    model_messages[-1] = {
                        "role": "user",
                        "content": user_input
                    }

                inputs = tokenizer.apply_chat_template(
                    model_messages,
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
                    response = "I couldn't generate a response."

                st.markdown(response)

        # Save assistant response
        st.session_state.messages.append(
            {"role": "assistant", "content": response}
        )
