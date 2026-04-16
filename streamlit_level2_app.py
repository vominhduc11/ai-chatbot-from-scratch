import time

import streamlit as st

from src.generative_inference import GenerativeChatbotEngine

st.set_page_config(
    page_title="AI Chatbot Level 2",
    page_icon="🤖",
    layout="centered",
)


@st.cache_resource
def load_engine():
    return GenerativeChatbotEngine()


engine = load_engine()

st.title("🤖 AI Chatbot Cấp 2")
st.caption("Seq2Seq generative chatbot. Model tự sinh câu trả lời theo từng token.")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Chào bạn. Đây là phiên bản cấp 2, nên mình sẽ tự sinh câu trả lời thay vì chỉ chọn câu mẫu cố định.",
        }
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input = st.chat_input("Nhập câu hỏi của bạn...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    result = engine.reply(user_input)
    response_text = result["answer"].strip()
    meta_lines = [
        f"**Mode:** `{result['mode']}`",
        f"**Overlap:** `{result['overlap_score']}`",
    ]
    if result.get("matched_input"):
        meta_lines.append(f"**Matched input:** `{result['matched_input']}`")
    full_text = response_text + "\n\n" + "  \n".join(meta_lines)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        streamed = ""
        words = response_text.split()
        for word in words:
            streamed += word + " "
            placeholder.markdown(streamed.strip())
            time.sleep(0.02)
        placeholder.markdown(full_text)

    st.session_state.messages.append({"role": "assistant", "content": full_text})

with st.sidebar:
    st.header("Thông tin model")
    st.write("Model: Seq2Seq GRU")
    st.write("Loại: Generative chatbot cấp 2")
    st.write("Training: From scratch từ dữ liệu input-output")
    st.write("Fallback: lexical overlap + retrieved fallback")

    if st.button("Xóa lịch sử chat"):
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Lịch sử chat đã được xóa. Hỏi tiếp đi.",
            }
        ]
        st.rerun()
