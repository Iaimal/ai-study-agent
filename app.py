# app.py
import streamlit as st
from agent import run_agent

st.set_page_config(page_title="AI Study Assistant", page_icon="🤖")

st.title("🤖 AI Study Assistant")
st.caption("Ask me about your notes, ask me to calculate something, or ask what day it is — I'll pick the right tool automatically.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg.get("tools_used"):
            st.caption(f"🔧 Used: {', '.join(msg['tools_used'])}")

question = st.chat_input("Ask me anything...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer, tools_used = run_agent(question)
        st.write(answer)
        if tools_used:
            st.caption(f"🔧 Used: {', '.join(tools_used)}")

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "tools_used": tools_used
    })