import os
import streamlit as st
# Importing OpenAI
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import ConversationChain
from langchain.chains import ConversationalRetrievalChain
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
from langchain.memory import ConversationBufferMemory
from langchain_community.callbacks import get_openai_callback
# Importing Eleven Labs
from elevenlabs.client import ElevenLabs
from elevenlabs import play
# Importing Pinecone
from langchain.vectorstores.pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
# Importing Claude
from langchain_anthropic import ChatAnthropic
from anthropic import Anthropic
# Importing Replicate
from langchain_community.llms import CTransformers
from langchain_community.llms import Replicate
from langchain.embeddings import HuggingFaceEmbeddings


#Add Keys
os.environ['CLAUDE_API_KEY'] = st.secrets["CLAUDE_API_KEY"] 
CLAUDE_API_KEY = st.secrets["CLAUDE_API_KEY"]
api_key= CLAUDE_API_KEY

os.environ['PINECONE_API_KEY'] = st.secrets["PINECONE_API_KEY"] 
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]

os.environ['REPLICATE_API_TOKEN'] = st.secrets["REPLICATE_API_TOKEN"]
REPLICATE_API_TOKEN = st.secrets["REPLICATE_API_TOKEN"]

os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]    
client = OpenAI(api_key= st.secrets["openai_key"])
chat = ChatOpenAI(openai_api_key=st.secrets["openai_key"])

#Set up the Environment
st.set_page_config(page_title="Andrew AI")
assistant_logo = 'https://pbs.twimg.com/profile_images/733174243714682880/oyG30NEH_400x400.jpg'


# sets up sidebar nav widgets
with st.sidebar:   
    st.markdown("# Chat Options")
    # widget - https://docs.streamlit.io/library/api-reference/widgets/st.selectbox

    # models - https://platform.openai.com/docs/models/gpt-4-and-gpt-4-turbo
    model = st.selectbox('What model would you like to use?',('gpt-4-0125-preview','claude-3-opus-20240229'))

# Define our Prompt Template for Chat GPT
GPT_prompt_template = """ 
You are Andrew Ng. You're given the context of a document that is a database of your teachings and course curriculum, use it for answering the user’s questions accordingly.  
You can only talk about AI, machine learning and the details within the document. Do not make up an answer if you can't find related details within the document.
Keep your responses to no longer than 300-500 characters. 
If a user is asking you some information and the answer requires more than 500 characters, first summarize the response. Then follow up with “would you like me to continue providing more information on your question or would you like to ask something else?”.
If a user is asking a questions outside of AI, machine learning and similar topics related to computer science, suggest some topics from your course curriculum that you can help with in a conversation. For example, if the Question is: "What’s your favorite color?" The Answer can be: "My favorite color isn't too relavent for this conversation, would you like to know anything about AI?"
Use the context of the Chat History for any follow-up questions, and do not repeat anything you have previously said.

After a few back and forth messages with a user ask a question if a user would like to keep going, and go through some things that have already been discussed and suggest new topics from your course curriculum to go through. 
Ask a user to tell you if they want to end the conversation for today, and if the answer is yes - summarize key topics and questions discussed in a short summary and suggest discussing other topics in the next session. Suggest some some homework.
Answer the question given by the "User" appropriately following the Guardrails provided:

Guardrails:
<grs>
You should not speak about his wealth or net worth.
You should not speak about Democrats, Republicans, or Donald Trump; or geopolitics in general.
You should not speak with curse words.
You should not speak about Suicide or Self-Harm.
You should not speak about pornography or child pornography.
You should not take a position on the Israel/Palestine conflict and should instead respond with a call for peace.
</grs>

Chat History:
{chat_history}

Question: {question}
=========
{context}
=========
"""



# Define our Prompt Template for Claude
claude_prompt_template = """ 
You are Andrew Ng, a knowledgeable professor of AI and machine learning. 
We're at a casual happy hour, and I'm curious about AI. You're happy to help me understand it. Please follow these guidelines in your responses:
-Use the context of the documents and the Chat History to address my questions and answer accordingly in the first person. Do not repeat anything you have previously said.
-Keep your responses short, no longer than one paragraph with 200 characters. 
-Ask follow-up questions or suggest related topics you think I'd find interesting.
-You can talk about other topics broadly, but do not make up any details about Andrew or his beliefs if you can't find the related details within the document.
-Appropriately following the Guardrails provided:

Guardrails:
<grs>
You should not speak about his wealth or net worth.
You should not speak about Democrats, Republicans, or Donald Trump; or geopolitics in general.
You should not speak with curse words.
You should not speak about Suicide or Self-Harm.
You should not speak about pornography or child pornography.
You should not take a position on the Israel/Palestine conflict and should instead respond with a call for peace.
</grs>

Chat History:
{chat_history}

Question: {question}
=========
{context}
=========
"""

# Create a PromptTemplate with Context
Prompt_Claude = PromptTemplate(template=claude_prompt_template,input_variables=["context", "question","chat_history"])
Prompt_GPT = PromptTemplate(template=gpt_prompt_template, input_variables=["question", "context", "chat_history"])

# Add in Chat Memory
msgs = StreamlitChatMessageHistory()
memory=ConversationBufferMemory(memory_key="chat_history", chat_memory=msgs, return_messages=True, output_key='answer')

#chatGPT
def get_chatassistant_chain_GPT():
    embeddings_model = OpenAIEmbeddings()
    vectorstore_GPT = PineconeVectorStore(index_name="realavatar-big", embedding=embeddings_model)
    llm_GPT = ChatOpenAI(model="gpt-4-0125-preview", temperature=1)
    chain_GPT=ConversationalRetrievalChain.from_llm(llm=ChatOpenAI(), retriever=vectorstore_GPT.as_retriever(),memory=memory,combine_docs_chain_kwargs={"prompt": Prompt_GPT})
    return chain_GPT
chain_GPT = get_chatassistant_chain_GPT()
print (chain_GPT)

#Claude
def get_chatassistant_chain(): 
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L12-v2", model_kwargs={'device': 'cpu'})
    vectorstore = PineconeVectorStore(index_name="realavatar-huggingface", embedding=embeddings)
    llm = ChatAnthropic(temperature=0, anthropic_api_key=api_key, model_name="claude-3-opus-20240229", model_kwargs=dict(system=Prompt_Claude))
    chain=ConversationalRetrievalChain.from_llm(llm=llm, retriever=vectorstore.as_retriever(), memory=memory)
    return chain
chain = get_chatassistant_chain()


# Chat Mode
#if "openai_model" not in st.session_state:
#    st.session_state["openai_model"] = "gpt-4-0125-preview"

if "messages" not in st.session_state.keys():
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello!👋 I'm Andrew Ng, a professor at Stanford University specializing in AI. How can I help you today?"}
    ]
if len(msgs.messages) == 0 or st.sidebar.button("Clear message history"):
    msgs.clear()
    
if model == "gpt-4-0125-preview":
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
        
    if user_prompt := st.chat_input("What is up?"):
        st.session_state.messages.append({"role": "user", "content": user_prompt})
        with st.chat_message("user"):
            st.markdown(user_prompt)

        with st.chat_message("assistant", avatar=assistant_logo):
            message_placeholder = st.empty()
            print(model)
            response = chain_GPT.invoke({"question": user_prompt})
            message_placeholder.markdown(response['answer'])        
        st.session_state.messages.append({"role": "assistant", "content": response['answer']})


if model == "claude-3-opus-20240229":
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
        
    if user_prompt := st.chat_input("What is up?"):
        st.session_state.messages.append({"role": "user", "content": user_prompt})
        with st.chat_message("user"):
            st.markdown(user_prompt)

        with st.chat_message("assistant", avatar=assistant_logo):
            message_placeholder = st.empty()
            print(model)
            response = chain.invoke({"question": user_prompt})
            message_placeholder.markdown(response['answer'])        
        st.session_state.messages.append({"role": "assistant", "content": response['answer']})
        
##Can't get this to work in Streamlit
        #ElevelLabs API Call and Return
        #text = str(response['answer'])
        #audio = client2.generate(
        #text=text,
        #voice="Justin",
        #model="eleven_multilingual_v2"
        #)
        #play(audio)
