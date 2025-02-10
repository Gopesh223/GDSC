import streamlit as st
import os
import pyttsx3
import faiss
from langchain.agents import initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain.tools import Tool
from streamlit_mic_recorder import speech_to_text
from duckduckgo_search import DDGS

pdf_path = "hp.pdf"
vectorstore_dir = "faiss_index"

embeddings = OllamaEmbeddings(model='llama3.2')
llm = OllamaLLM(model='llama3.2')

os.makedirs(vectorstore_dir, exist_ok=True)

if os.path.exists(os.path.join(vectorstore_dir, "index.faiss")):
    vector_store = FAISS.load_local(vectorstore_dir, embeddings, allow_dangerous_deserialization=True)
else:
    loader = PDFPlumberLoader(pdf_path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunked_documents = text_splitter.split_documents(documents)

    vector_store = FAISS.from_documents(chunked_documents, embeddings)
    vector_store.save_local(vectorstore_dir)

def retrieve_docs(query):
    docs = vector_store.similarity_search(query)
    return docs if docs else None

def answer_question(question, documents):
    if not documents:
        return "I couldn't find relevant information in the PDF."

    context = "\n\n".join([doc.page_content for doc in documents])
    template = """
    You are an intelligent assistant. Use the retrieved context to answer the question.
    If uncertain, ask the user for clarification.
    
    Question: {question}
    Context: {context}
    
    Answer:
    """
    return llm.invoke(template.format(question=question, context=context))

def search_web(query):
    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=1))
    return results[0]["body"] if results else "No relevant web results found."

def text_to_speech(response):
    engine = pyttsx3.init()
    engine.say(response)
    engine.runAndWait()

tools = [
    Tool(name="FAISS Search", func=retrieve_docs, description="Retrieves documents from FAISS."),
    Tool(name="Web Search", func=search_web, description="Fetches search results from DuckDuckGo."),
]

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    memory=memory,
)

st.title("Sorcerer's Stone Chatbot")

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

def run_agentic_query(question):
    st.chat_message("user").write(question)

    related_documents = retrieve_docs(question)
    if related_documents:
        answer = answer_question(question, related_documents)
    else:
        st.write("No relevant content in FAISS. Searching the web...")
        answer = search_web(question)

    st.chat_message("assistant").write(answer)
    text_to_speech(answer)

if st.button("Use Speech Input"):
    if state.text_received:
        run_agentic_query(state.text_received[-1])

question = st.chat_input("Ask your question")
if question:
    run_agentic_query(question)
