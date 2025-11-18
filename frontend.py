# frontend.py
import streamlit as st
import requests

st.title("AI Chatbot")

user_input = st.text_input("Ask me anything:")

if st.button("Send"):
    response = requests.post(
        "http://localhost:5000/chat",  # Backend on EC2 localhost:5000
        json={"user_message": user_input}
    )
    if response.ok:
        result = response.json()
        st.write("Response:", result['llm_reply'])
    else:
        st.write("Error:", response.text)
