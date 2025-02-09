import streamlit as st
import os
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama.llms import OllamaLLM
from streamlit_mic_recorder import speech_to_text

pdf_path = "hp.pdf"

model = OllamaLLM(model='llama3.2')

def load_pdf():
    loader = PDFPlumberLoader(pdf_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=500)
    chunked_documents = text_splitter.split_documents(documents)
    return chunked_documents

documents = load_pdf()

def answer_question(question):
    context = "\n\n".join([doc.page_content for doc in documents])
    template = """
    You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question.
    If you don't know the answer, just say that you don't know.
    
    Question: {question} 
    Context: {context} 
    
    Answer:
    """
    return model.invoke({"question": question, "context": context})

st.title("The Sorcerer's Stone Chatbot")

state = st.session_state
if 'text_received' not in state:
    state.text_received = []

c1, c2 = st.columns(2)
with c1:
    st.write("Convert speech to text:")
with c2:
    text = speech_to_text(language='en', use_container_width=True, just_once=True, key='STT')

if text:
    state.text_received.append(text)

def run_query(question):
    if question:
        st.chat_message("user").write(question)
        answer = answer_question(question)
        st.chat_message("assistant").write(answer)

if st.button("Use Speech Input"):
    if state.text_received:
        run_query(state.text_received[-1])

question = st.chat_input("Ask your question")
if question:
    run_query(question)
