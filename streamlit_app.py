import streamlit as st
import os
import tempfile
from gtts import gTTS
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from streamlit_mic_recorder import speech_to_text

# File paths
pdf_path = "hp.pdf"
vectorstore_dir = tempfile.mkdtemp()  # Creates a temporary directory

# Initialize models
embeddings = OllamaEmbeddings(model='llama3.2')
model = OllamaLLM(model='llama3.2')

# Load or create FAISS index
if os.path.exists(os.path.join(vectorstore_dir, "index.faiss")):
    vector_store = FAISS.load_local(vectorstore_dir, embeddings, allow_dangerous_deserialization=True)
else:
    loader = PDFPlumberLoader(pdf_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=500)
    chunked_documents = text_splitter.split_documents(documents)
    vector_store = FAISS.from_documents(chunked_documents, embeddings)
    vector_store.save_local(vectorstore_dir)

# Retrieve relevant documents
def retrieve_docs(query):
    return vector_store.similarity_search(query)

# Generate answer using model
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

# Text-to-Speech Function
def text_to_speech(text):
    tts = gTTS(text=text, lang='en')
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio:
        tts.save(temp_audio.name)
        return temp_audio.name

# Streamlit UI
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
        
        # Play answer as speech
        audio_file = text_to_speech(answer)
        st.audio(audio_file, format="audio/mp3")

if st.button("Use Speech Input"):
    if state.text_received:
        run_query(state.text_received[-1])

question = st.chat_input("Ask your question")
if question:
    run_query(question)
