import streamlit as st
from llama_index import VectorStoreIndex, ServiceContext, Document
from llama_index.llms import OpenAI
import openai
from llama_index import SimpleDirectoryReader
import os
import tempfile

st.set_page_config(page_title="Chat with your syllabus!", page_icon="🦙", layout="centered", initial_sidebar_state="auto", menu_items=None)
openai.api_key = st.secrets.openai_key
st.title("Chat with the Syllabus and Course Outline 💬🦙")
st.info("Check out the Data Analytics Program at Miami Dade College [MDC](https://mdc.edu)", icon="📃")

uploaded_file = st.sidebar.file_uploader("Upload a document", type=["pdf", "docx", "txt"])

if uploaded_file:
    file_dir = 'data'
    os.makedirs(file_dir, exist_ok=True)  # Ensure the directory exists
    file_path = os.path.join(file_dir, "uploaded_file.txt")
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getvalue())
    input_dir = file_dir
    st.write(f'File saved at: {file_path}')  # Debugging line
else:
    input_dir = "./data"


st.write(f'File saved at: {file_path}')  # Debugging line

@st.cache_resource(show_spinner=False)
def load_data(input_dir):
    with st.spinner(text="Loading and indexing the course docs – hang tight! This should take 1-2 minutes."):
        reader = SimpleDirectoryReader(input_dir=input_dir, recursive=True)
        docs = reader.load_data()
        st.write(f'Loaded docs: {docs}')  # Debugging line
        service_context = ServiceContext.from_defaults(llm=OpenAI(model="gpt-3.5-turbo", temperature=0.5, system_prompt="You are a College Professor or Corporate Trainer and your job is to answer questions about the course. Assume that all questions are related to the course and documents provided. Keep your answers technical and based on facts – do not hallucinate features."))
        index = VectorStoreIndex.from_documents(docs, service_context=service_context)
        return index

index = load_data(input_dir)
chat_engine = index.as_chat_engine(chat_mode="condense_question", verbose=True)

if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "Ask me a question about the course!"}]

if prompt := st.chat_input("Your question"):
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = chat_engine.chat(prompt)
            st.write(response.response)
            message = {"role": "assistant", "content": response.response}
            st.session_state.messages.append(message)
