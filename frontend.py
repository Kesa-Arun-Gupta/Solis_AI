import streamlit as st
import requests
import pandas as pd

st.set_page_config(page_title="SOLIS AI Chatbot", layout="wide")

st.title("[translate:SOLIS AI] - Conversational Analytics")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

def render_chat_messages():
    for i, chat in enumerate(st.session_state.chat_history):
        if chat["sender"] == "user":
            with st.chat_message("user", key=f"user_{i}"):
                st.markdown(chat["text"])
        else:
            with st.chat_message("assistant", key=f"assistant_{i}"):
                # If assistant message contains table data, display table or chart
                if "table_data" in chat:
                    df = pd.DataFrame(chat["table_data"]["rows"], columns=chat["table_data"]["columns"])
                    st.dataframe(df)
                else:
                    st.markdown(chat["text"])

def send_message():
    user_msg = st.session_state.input_text.strip()
    if not user_msg:
        st.warning("Please enter a question.")
        return
    
    # Append user message to chat
    st.session_state.chat_history.append({"sender": "user", "text": user_msg})
    render_chat_messages()

    # Call backend API
    try:
        response = requests.post("http://localhost:5000/chat", json={"user_message": user_msg}, timeout=30)
        if response.ok:
            data = response.json()

            # Append assistant answer text and table/chart data
            st.session_state.chat_history.append({
                "sender": "assistant",
                "text": data.get("answer", "No answer."),
                "table_data": data.get("table_data")
            })
            render_chat_messages()

            # Show recommended questions as buttons
            rec_qs = data.get("recommended_questions", [])
            if rec_qs:
                st.markdown("### Suggested questions")
                for q in rec_qs:
                    if st.button(q):
                        st.session_state.input_text = q
                        send_message()
        else:
            st.error(f"Error from backend: {response.text}")
    except Exception as e:
        st.error(f"Error contacting backend: {e}")

st.text_input("Ask your question here", key="input_text", on_change=send_message)

render_chat_messages()
