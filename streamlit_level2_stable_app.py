import streamlit as st

from src.generative_inference_v2 import GenerativeChatbotEngineV2

st.set_page_config(
    page_title="AI Chatbot Level 2 Stable",
    page_icon="🤖",
    layout="centered",
)


def load_engine():
    return GenerativeChatbotEngineV2()


if "engine" not in st.session_state:
    st.session_state.engine = load_engine()

engine = st.session_state.engine

st.title("🤖 AI Chatbot Cấp 2 Ổn Định")
st.caption("Bản vá ưu tiên exact match, siết fallback và tránh câu sinh generic cho mọi input.")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Chào bạn. Đây là bản vá ổn định hơn của chatbot cấp 2. Nếu model sinh câu quá dở, hệ thống sẽ fallback an toàn.",
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
    meta_lines = [
        f"**Mode:** `{result['mode']}`",
        f"**Overlap:** `{result['overlap_score']}`",
        f"**Exact match:** `{result['exact_match']}`",
    ]
    if result.get("matched_input"):
        meta_lines.append(f"**Matched input:** `{result['matched_input']}`")

    full_text = result["answer"].strip() + "\n\n" + "  \n".join(meta_lines)

    with st.chat_message("assistant"):
        st.markdown(full_text)

    st.session_state.messages.append({"role": "assistant", "content": full_text})

with st.sidebar:
    st.header("Thông tin model")
    st.write("Model: Seq2Seq GRU")
    st.write("Inference: exact match + lexical overlap + safe fallback")
    st.write("Artifacts: level2_stable_model.pt / level2_stable_metadata.json")

    if st.button("Reload model"):
        st.session_state.engine = load_engine()
        st.rerun()

    if st.button("Xóa lịch sử chat"):
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Lịch sử chat đã được xóa. Hỏi tiếp đi.",
            }
        ]
        st.rerun()
