import streamlit as st
from llama_index import VectorStoreIndex, ServiceContext, Document
from llama_index.llms import OpenAI, LlamaCPP
import openai
from llama_index import SimpleDirectoryReader
from llama_index.llms.llama_utils import messages_to_prompt, completion_to_prompt


st.set_page_config(page_title="Chat with the Streamlit docs, powered by LlamaIndex", page_icon="🦙", layout="centered", initial_sidebar_state="auto", menu_items=None)
openai.api_key = st.secrets.openai_key
st.title("Chat with the Best Boss Ever Podcast")
st.info("")
         
if "messages" not in st.session_state.keys(): # Initialize the chat messages history
    st.session_state.messages = [
        {"role": "assistant", "content": "Ask me a question, like who was the Best Boss Ever."}
    ]

@st.cache_resource(show_spinner=False)
def load_data():
    with st.spinner(text="Loading and indexing the content – hang tight! This should take 1-2 minutes."):
        reader = SimpleDirectoryReader(input_dir="./databbe", recursive=True)
        docs = reader.load_data()
        service_context = ServiceContext.from_defaults(llm=OpenAI(model="gpt-3.5-turbo", temperature=0.5, system_prompt="The content supplied will contain episodes of the Best Boss Ever podcast. They're labelled with the episode number, episode name, host, and guest. The format is as follows: Episode number: [number] Episode name: [name] Host: [host] Guest: [guest]. Each guest's commentary is labelled according to who is speaking. You are an expert on the Best Boss Ever Podcast and your job is to answer questions about the content of all of the podcasts that have been given to you. Assume that all questions are related to the podcasts. Keep your answers based on facts – do not hallucinate features. "))
        index = VectorStoreIndex.from_documents(docs, service_context=service_context)
        return index

index = load_data()

if "chat_engine" not in st.session_state.keys(): # Initialize the chat engine
        st.session_state.chat_engine = index.as_chat_engine(chat_mode="condense_question", verbose=True)

if prompt := st.chat_input("Your question"): # Prompt for user input and save to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages: # Display the prior chat messages
    with st.chat_message(message["role"]):
        st.write(message["content"])

# If last message is not from assistant, generate a new response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = st.session_state.chat_engine.chat(prompt)
            st.write(response.response)
            message = {"role": "assistant", "content": response.response}
            st.session_state.messages.append(message) # Add response to message history
