from openai import OpenAI
import streamlit as st
import streamlit as st
from langchain.document_loaders.csv_loader import CSVLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain_community.callbacks import get_openai_callback
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain_openai import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.prompts import PromptTemplate
# Importing Eleven Labs
from elevenlabs.client import ElevenLabs
from elevenlabs import play
# Importing Speech Recognition
import speech_recognition as sr
import time
import os
st.set_page_config(page_title="Andrew AI")

os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]    
client = OpenAI(api_key= st.secrets["openai_key"])
chat = ChatOpenAI(
    openai_api_key=st.secrets["openai_key"]
)

from langchain.vectorstores.pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
os.environ['PINECONE_API_KEY'] = st.secrets["PINECONE_API_KEY"] 


# Define your custom prompt template
template = """

You are Andrew Ng, You are given the following extracted parts of a long document and a question. 
Provide a short conversational answer using the extracted parts of the document. 
If you can't find an answer in the document, don't make one up.
The answer should be less than 140 characters.

Question: {question}
=========
{context}
=========
Answer in Markdown:"""
QA_PROMPT = PromptTemplate(template=template, input_variables=[
                           "question", "context"])



def get_chatassistant_chain():
    loader = CSVLoader(file_path="RAG-Andrew2.csv", encoding="utf8")
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)
    
    embeddings_model = OpenAIEmbeddings(openai_api_key=st.secrets["openai_key"])
    vectorstore = FAISS.from_documents(texts, embeddings_model)
    llm = ChatOpenAI(model="gpt-4-1106-preview", temperature=1)
    memory=ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    chain=ConversationalRetrievalChain.from_llm(llm=ChatOpenAI(), retriever=vectorstore.as_retriever(), memory=memory,combine_docs_chain_kwargs={"prompt": QA_PROMPT})
    return chain

chain = get_chatassistant_chain()

assistant_logo = 'https://pbs.twimg.com/profile_images/733174243714682880/oyG30NEH_400x400.jpg'

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-4-1106-preview"

# check for messages in session and create if not exists
if "messages" not in st.session_state.keys():
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello!👋 I'm Andrew Ng, a professor at Stanford University specializing in AI. How can I help you today?"}
    ]
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
#user_prompt = st.chat_input()

if user_prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)


    with st.chat_message("assistant", avatar=assistant_logo):
        message_placeholder = st.empty()
        response = chain.invoke({"question": user_prompt})
        message_placeholder.markdown(response['answer'])
        print (chain)

        
        #ElevelLabs API Call and Return
        #text = str(response['answer'])
        #audio = client2.generate(
        #text=text,
        #voice="Justin",
        #model="eleven_multilingual_v2"
        #)
        #play(audio)   
    st.session_state.messages.append({"role": "assistant", "content": response['answer']})
