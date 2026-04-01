import streamlit as st
from src.inference import ChatbotEngine

st.set_page_config(
    page_title="AI Chatbot From Scratch",
    page_icon="🤖",
    layout="centered"
)

@st.cache_resource
def load_engine():
    return ChatbotEngine()

engine = load_engine()

st.title("🤖 AI Chatbot Tự Train Từ Đầu")
st.caption("Model tự xây bằng PyTorch, không dùng pretrained model")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Chào bạn. Hỏi mình về AI, Python, machine learning hoặc project."
        }
    ]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Nhập câu hỏi của bạn...")

if user_input:
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    result = engine.reply(user_input)

    bot_text = (
        f"{result['answer']}\n\n"
        f"**Intent:** `{result['intent']}`  \n"
        f"**Confidence:** `{result['confidence']}`"
    )

    st.session_state.messages.append({
        "role": "assistant",
        "content": bot_text
    })

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        st.markdown(bot_text)

with st.sidebar:
    st.header("Thông tin model")
    st.write("Model: Embedding + Mean Pooling + MLP")
    st.write("Loại: Intent Classification Chatbot")
    st.write("Training: From scratch")
    st.write("Pretrained model: Không dùng")

    if st.button("Xóa lịch sử chat"):
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Lịch sử chat đã được xóa. Hỏi tiếp đi."
            }
        ]
        st.rerun()