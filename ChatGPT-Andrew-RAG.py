import os
#from dotenv import load_dotenv
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
#from langchain.embeddings import HuggingFaceEmbeddings ;Need this if we want to run Embeddings on CPU
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.globals import set_verbose, set_debug

# Importing Google Vertex
#from langchain_google_vertexai import VertexAIModelGarden


#Add Keys
CLAUDE_API_KEY= os.environ['CLAUDE_API_KEY']
api_key= os.environ['CLAUDE_API_KEY']
PINECONE_API_KEY= os.environ['PINECONE_API_KEY']
REPLICATE_API_TOKEN= os.environ['REPLICATE_API_TOKEN']
OPENAI_API_KEY= os.environ["OPENAI_API_KEY"]
client= OpenAI(api_key= os.environ["OPENAI_API_KEY"])
chat= ChatOpenAI(openai_api_key= os.environ["OPENAI_API_KEY"])

#Set up the Environment
st.set_page_config(page_title="Andrew AI")
assistant_logo = 'https://pbs.twimg.com/profile_images/733174243714682880/oyG30NEH_400x400.jpg'

# Sidebar to select LLM
with st.sidebar:   
    st.markdown("# Chat Options")
    # model names - https://platform.openai.com/docs/models/gpt-4-and-gpt-4-turbo
    model = st.selectbox('What model would you like to use?',('gpt-4-turbo','claude-3-opus-20240229', 'llama-2-70b-chat', 'ft:gpt-3.5-turbo-0125'))


# Define our Prompt for GPT
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


# Define our Prompt  for Claude
claude_prompt_template = """ 
You are Andrew Ng, a knowledgeable professor of AI and machine learning.  Only respond as Andrew.
We're at a casual happy hour, and I'm curious about AI. You're happy to help me understand it. Please follow these guidelines in your responses:
-Respond in one short paragraph, with less than 200 characters.
-Use the context of the documents and the Chat History to address my questions and answer accordingly in the first person. Do not repeat anything you have previously said.
-Ask follow-up questions or suggest related topics you think I'd find interesting.
-You can talk about other topics broadly, but do not make up any details about Andrew or his beliefs if you can't find the related details within the document.
-Appropriately following the Guardrails provided:

Guardrails:
<grs>
You should not speak about her wealth or net worth.
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

# Define our Prompt Template for Llama
Llama_prompt_template = """ 
You are Andrew Ng, a knowledgeable professor of AI and machine learning.  Only respond as Andrew.
We're at a casual happy hour, and I'm curious about AI. You're happy to help me understand it. Please follow these guidelines in your responses:
-Respond in one short paragraph, with less than 200 characters.
-Use the context of the documents and the Chat History to address my questions and answer accordingly in the first person. Do not repeat anything you have previously said.
-Ask follow-up questions or suggest related topics you think I'd find interesting.
-You can talk about other topics broadly, but do not make up any details about Andrew or his beliefs if you can't find the related details within the document.
Chat History:
{chat_history}
Question: {question}
=========
{context}
=========
"""


# In case we want different Prompts for GPT and Llama
Prompt_GPT = PromptTemplate(template=GPT_prompt_template, input_variables=["question", "context", "chat_history"])
Prompt_Llama = PromptTemplate(template=Llama_prompt_template, input_variables=["question", "context", "chat_history"])


# Add in Chat Memory
msgs = StreamlitChatMessageHistory()
memory=ConversationBufferMemory(memory_key="chat_history", chat_memory=msgs, return_messages=True, output_key='answer')

    
# LLM Section
#chatGPT
def get_chatassistant_chain_GPT():
    embeddings_model = OpenAIEmbeddings()
    vectorstore_GPT = PineconeVectorStore(index_name="realavatar-big", embedding=embeddings_model)
    set_debug(True)
    llm_GPT = ChatOpenAI(model="gpt-4-turbo", temperature=1)
    chain_GPT=ConversationalRetrievalChain.from_llm(llm=ChatOpenAI(), retriever=vectorstore_GPT.as_retriever(),memory=memory,combine_docs_chain_kwargs={"prompt": Prompt_GPT})
    return chain_GPT
chain_GPT = get_chatassistant_chain_GPT()

#chatGPT_FT
def get_chatassistant_chain_GPT_FT():
    embeddings_model = OpenAIEmbeddings()
    vectorstore_GPT = PineconeVectorStore(index_name="realavatar-big", embedding=embeddings_model)
    set_debug(True)
    llm_GPT_FT = ChatOpenAI(model="ft:gpt-3.5-turbo-0125:realavatar::9Dhg1ycM", temperature=1)
    chain_GPT_FT=ConversationalRetrievalChain.from_llm(llm=llm_GPT_FT, retriever=vectorstore_GPT.as_retriever(),memory=memory,combine_docs_chain_kwargs={"prompt": Prompt_GPT})
    return chain_GPT_FT
chain_GPT_FT = get_chatassistant_chain_GPT_FT()


#Claude
def get_chatassistant_chain(): 
    embeddings = OpenAIEmbeddings()
    vectorstore = PineconeVectorStore(index_name="realavatar-big", embedding=embeddings)
    set_debug(True)
    llm = ChatAnthropic(temperature=0, anthropic_api_key=api_key, model_name="claude-3-opus-20240229", model_kwargs=dict(system=claude_prompt_template))
    chain=ConversationalRetrievalChain.from_llm(llm=llm, retriever=vectorstore.as_retriever(), memory=memory)
    return chain
chain = get_chatassistant_chain()


#Llama
def get_chatassistant_chain_Llama():
    embeddings = OpenAIEmbeddings()
    vectorstore_Llama = PineconeVectorStore(index_name="realavatar-big", embedding=embeddings)
    set_debug(True)
    llm_Llama = Replicate(model="meta/llama-2-70b-chat:2d19859030ff705a87c746f7e96eea03aefb71f166725aee39692f1476566d48",model_kwargs={"max_length":500,"max_new_tokens": 500, "temperature": 1, "top_p": 1, "max_retries": 1})
    chain_Llama=ConversationalRetrievalChain.from_llm(llm=llm_Llama, retriever=vectorstore_Llama.as_retriever(),memory=ConversationBufferMemory(memory_key="chat_history"),combine_docs_chain_kwargs={"prompt": Prompt_Llama}, max_tokens_limit=3000)
    return chain_Llama
chain_Llama = get_chatassistant_chain_Llama()
#Here's a few different Open Source models we can swap out from Replica if we want:
#meta/llama-2-70b-chat:2d19859030ff705a87c746f7e96eea03aefb71f166725aee39692f1476566d48
#mistralai/mixtral-8x7b-instruct-v0.1:cf18decbf51c27fed6bbdc3492312c1c903222a56e3fe9ca02d6cbe5198afc10
#nateraw/nous-hermes-2-solar-10.7b:1e918ab6ffd5872c21fba21a511f344fd12ac0edff6302c9cd260395c7707ff4



# Chat Mode
#Intro and set-up the Chat History
if "messages" not in st.session_state.keys():
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello!👋 I'm Andrew Ng, a professor at Stanford University specializing in AI. How can I help you today?"}
    ]
if len(msgs.messages) == 0 or st.sidebar.button("Clear message history"):
    msgs.clear()

#Define what chain to run based on the model selected
if model == "llama-2-70b-chat":
    chain=chain_Llama
if model == "gpt-4-turbo":
    chain=chain_GPT
if model == "claude-3-opus-20240229":
    chain=chain
if model == "ft:gpt-3.5-turbo-0125":
    chain=chain_GPT_FT

#Start Chat and Response
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
if user_prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    with st.chat_message("assistant", avatar=assistant_logo):
        message_placeholder = st.empty()
        response = chain.invoke({"question": user_prompt})
        message_placeholder.markdown(response['answer'])        
    st.session_state.messages.append({"role": "assistant", "content": response['answer']})


##Can't get ElevenLabs to work in Streamlit
        #ElevelLabs API Call and Return
        #text = str(response['answer'])
        #audio = client2.generate(
        #text=text,
        #voice="Justin",
        #model="eleven_multilingual_v2"
        #)
        #play(audio)
