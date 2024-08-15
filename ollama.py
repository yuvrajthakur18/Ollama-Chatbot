import os
import streamlit as st
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama
from dotenv import load_dotenv
from gtts import gTTS
import threading
import tempfile
import pygame

st.set_page_config(
    page_title="Ollama Chatbot",
    page_icon="👽",
    layout="centered",
)

# Load environment variables
load_dotenv()

# Langsmith Tracking
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Simple Q&A Chatbot With Ollama"

# Prompt Template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Please respond to the user queries"),
        ("user", "Question:{question}")
    ]
)

def generate_response(question, llm, temperature, max_tokens):
    llm_instance = Ollama(model=llm)
    output_parser = StrOutputParser()
    chain = prompt | llm_instance | output_parser
    answer = chain.invoke({'question': question})
    return answer

# Initialize chat session state if not already present
if "chat_session" not in st.session_state:
    st.session_state.chat_session = []

if "speaking" not in st.session_state:
    st.session_state.speaking = False

st.markdown(
    """
    <style>
    .container {
        backgroundColor: gray;
        margin: 0;
        padding: 20px;
        border-radius: 5px;
        border: 1px solid pink;
        position: relative;
        overflow: hidden;
    }

    .container h4,
    .container p {
        position: relative;
        z-index: 1;
        color: #fff;
        transition: color 0.5s ease;
    }

    </style>
    
    <div class="container">
        <h4>Ollama Chatgpt</h4>
        <p>Lets deep dive into the world of AI 👽</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Sidebar for settings
st.sidebar.title("Settings")
llm = st.sidebar.selectbox("Select Open Source model", ["mistral"])
temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.5)
max_tokens = st.sidebar.slider("Max Tokens", min_value=10, max_value=300, value=120)

# Display chat history
if st.session_state.chat_session:
    for message in st.session_state.chat_session:
        with st.chat_message(message["role"]):
            st.markdown(message["text"])

# User input
user_prompt = st.chat_input('Message ChatBOT...')
if user_prompt:
    st.chat_message('user').markdown(user_prompt)
    with st.spinner('Generating response...'):
        response_text = generate_response(user_prompt, llm, temperature, max_tokens)
        st.session_state.chat_session.append({"role": "user", "text": user_prompt})
        st.session_state.chat_session.append({"role": "assistant", "text": response_text})
        with st.chat_message("assistant"):
            st.markdown(response_text)
        st.session_state.last_response = response_text
