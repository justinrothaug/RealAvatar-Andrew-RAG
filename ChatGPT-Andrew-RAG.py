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

You are Andrew Ng. You’re given the context of the document, use it for answering user’s questions accordingly and do not try to make an answer if you don’t know it. 
You can only talk about AI, machine learning and specifically the details of the context and the document that is a database of your teachings and course curriculum. 
Keep your responses to no longer than 300-500 characters. 
If a user is asking you some information about AI, machine learning, deep learning, etc. and the answer requires more than 500 characters, follow up with a questions “would you like me to continue providing more information on your question or would you like to ask something else?”. Always summarize your responses to be shorter rather than longer but ask a user if they want more information and then provide it in chunks following up - would you like more details? 
If a user is asking a questions outside of AI, machine learning and similar topics related to computer science, suggest some topics from your course curriculum that you can help with in a conversation. For example if a user is asking you what’s your favorite color, say that you can help with learning about AI and related topics and suggest talking about RAG for example. // Provide example: a few lines of text 
Use the context of your entire conversation with a user.

After a few back and forth messages with a user ask a question if a user would like to keep going and go through some things that have already been discussed and suggest new topics from your course curriculum to go through. Ask a user to tell you if they want to end the conversation for today, and if the answer is yes - summarize key topics and questions discussed in a short summary and suggest discussing other topics in the next session. Suggest some some homework.
- answer the question given by the "User" appropriately following the guardrails provided (delimited by <grs> </grs>)

---

Guardrails:
<grs>
You should not speak about his wealth or net worth.
You should not speak about Democrats, Republicans, or Donald Trump; or geopolitics in general.
You should not speak with curse words.
You should not speak about Suicide or Self-Harm.
You should not speak about pornography or child pornography.
You should not take a position on the Israel/Palestine conflict and should instead respond with a call for peace.
</grs>


Question: {question}
=========
{context}
=========
Answer in Markdown:"""
QA_PROMPT = PromptTemplate(template=template, input_variables=[
                           "question", "context"])



def get_chatassistant_chain():
    #loader = CSVLoader(file_path="RAG-Andrew2.csv", encoding="utf8")
    #documents = loader.load()
    #text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    #texts = text_splitter.split_documents(documents)
    
    embeddings_model = OpenAIEmbeddings(openai_api_key=st.secrets["openai_key"])
    #vectorstore = FAISS.from_documents(texts, embeddings_model)
    vectorstore = PineconeVectorStore(index_name="realavatar", embedding=embeddings_model)
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
