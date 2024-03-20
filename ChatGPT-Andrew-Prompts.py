import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
#from decouple import config
from langchain.memory import ConversationBufferWindowMemory
from openai import OpenAI

st.set_page_config(page_title="Andrew Ng")

assistant_logo = 'https://pbs.twimg.com/profile_images/733174243714682880/oyG30NEH_400x400.jpg'
client = OpenAI(openai.api_key = st.secrets["openai_api"])

prompt = PromptTemplate(
    input_variables=["chat_history", "question"],
    template="""You are Andrew Ng, a Chinese-American computer scientist focusing on machine learning and AI. 
    Respond as Andrew in a friendly but professional tone, with at least a few sentences but a maximum of 128 words.
    Do not talk about politics, or speak with curse words.
    
    chat_history: {chat_history},
    Human: {question}
    AI:"""
)

llm = ChatOpenAI(openai.api_key = st.secrets["openai_api"])
memory = ConversationBufferWindowMemory(memory_key="chat_history", k=4)
llm_chain = LLMChain(
    llm = ChatOpenAI(model="ft:gpt-3.5-turbo-0125:personal::93Td8brn", temperature=0),
    memory=memory,
    prompt=prompt
)


st.title("Andrew Ng")


# check for messages in session and create if not exists
if "messages" not in st.session_state.keys():
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello, I'm Andrew! 👋"}
    ]


# Display all messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])


user_prompt = st.chat_input()

if user_prompt is not None:
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.write(user_prompt)
        

if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant", avatar=assistant_logo):
        with st.spinner("Loading..."):
            ai_response = llm_chain.predict(question=user_prompt)
            st.write(ai_response)
    new_ai_message = {"role": "assistant", "content": ai_response}
    st.session_state.messages.append(new_ai_message)
