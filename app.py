import streamlit as st
from chatbot import ask_question

st.title("📚 Free Wikipedia RAG Chatbot")

query = st.text_input("Ask a question:")

if query:
    with st.spinner("Thinking..."):
        answer = ask_question(query)
    st.subheader("Answer")
    st.write(answer)
