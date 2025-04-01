import os
import streamlit as st
import requests
import agent_pricing

# The backend URL is now expected to point to the FastAPI agent.
BACKEND_URL = os.environ.get("BACKEND_URL") or "http://localhost:8000"

st.write("# Welcome to Media Streaming Assistant! 👋")
st.write(
    "Greetings! I'm your AI assistant here to help you configure your media streaming setup."
)

# Initialize chat history in session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display previous chat messages
for role, message in st.session_state.chat_history:
    with st.chat_message(role):
        st.markdown(message)

# Get new message from the user
prompt = st.chat_input("Type your message here...")

if prompt:
    # Display the user's message
    st.session_state.chat_history.append(("user", prompt))
    with st.chat_message("user"):
        st.markdown(prompt)

    # Call the FastAPI langgraph agent at the /chat/ endpoint.
    try:
        res = agent_pricing.chat(prompt)
        response_data = res.json()
        assistant_response = response_data.get("response", "No response received.")
    except Exception as e:
        assistant_response = f"Error calling agent: {e}"

    # Display the assistant's response and update chat history
    st.session_state.chat_history.append(("assistant", assistant_response))
    with st.chat_message("assistant"):
        st.markdown(assistant_response)