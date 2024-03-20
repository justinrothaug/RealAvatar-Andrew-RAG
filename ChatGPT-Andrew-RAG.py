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


def get_chatassistant_chain():
    loader = CSVLoader(file_path="./docs/DLScripts.csv")
    documents = loader.load()
    embeddings_model = OpenAIEmbeddings(openai_api_key= st.secrets["openai_key"])
    vectorstore = FAISS.from_documents(documents, embeddings_model)
    llm = ChatOpenAI(model="ft:gpt-3.5-turbo-0125:personal::93Td8brn", temperature=1)
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True) 
    chain = ConversationalRetrievalChain.from_llm(llm=ChatOpenAI(openai_api_key= st.secrets["openai_key"]),
                                                  retriever=vectorstore.as_retriever(),
                                                  memory=memory)                                      
    return chain


chain = get_chatassistant_chain()

assistant_logo = 'https://pbs.twimg.com/profile_images/733174243714682880/oyG30NEH_400x400.jpg'

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "ft:gpt-3.5-turbo-0125:personal::93Td8brn"

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "system", "content": "You are Andrew Ng, a Chinese-American computer scientist focusing on machine learning and AI. You have a wife named Carol, and two children. Respond to the following lines of dialog as Andrew Ng"}]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)


    with st.chat_message("assistant", avatar=assistant_logo):
        message_placeholder = st.empty()
        response = chain.invoke({"question": prompt})
        message_placeholder.markdown(response['answer'])

    st.session_state.messages.append({"role": "assistant", "content": response['answer']})
