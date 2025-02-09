import streamlit as st
import os
#import pyttsx3
#import faiss
#import pickle
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from streamlit_mic_recorder import speech_to_text

pdf_path = "hp.pdf"
vectorstore_dir = "faiss_index"

embeddings = OllamaEmbeddings(model='llama3.2')
model = OllamaLLM(model='llama3.2')

os.makedirs(vectorstore_dir, exist_ok=True)

if os.path.exists(os.path.join(vectorstore_dir, "index.faiss")):
    vector_store = FAISS.load_local(vectorstore_dir, embeddings, allow_dangerous_deserialization=True)
else:
    loader = PDFPlumberLoader(pdf_path)
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=500)
    chunked_documents = text_splitter.split_documents(documents)

    vector_store = FAISS.from_documents(chunked_documents, embeddings)

    vector_store.save_local(vectorstore_dir)

def retrieve_docs(query):
    return vector_store.similarity_search(query)

def answer_question(question, documents):
    context = "\n\n".join([doc.page_content for doc in documents])
    template = """
    You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question.
    If you don't know the answer, just say that you don't know.
    
    Question: {question} 
    Context: {context} 
    
    Answer:
    """
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | model
    return chain.invoke({"question": question, "context": context})

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
        related_documents = retrieve_docs(question)
        answer = answer_question(question, related_documents)
        st.chat_message("assistant").write(answer)
        #engine=pyttsx3.init()
        #engine.say(answer)
        #engine.runAndWait()

if st.button("Use Speech Input"):
    if state.text_received:
        run_query(state.text_received[-1])

question = st.chat_input("Ask your question")
if question:
    run_query(question)

