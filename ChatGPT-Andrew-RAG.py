import os
from dotenv import load_dotenv
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
from langchain_core.tracers.context import tracing_v2_enabled
# Importing Eleven Labs and HTML Audio
from elevenlabs.client import ElevenLabs
from elevenlabs import play
import base64
import array
# Importing Pinecone
from langchain.vectorstores.pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
# Importing Claude
from langchain_anthropic import ChatAnthropic
from anthropic import Anthropic
import re
# Importing Perplexity
from langchain_community.chat_models import ChatPerplexity
from langchain_core.prompts import ChatPromptTemplate
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.globals import set_verbose, set_debug
from streamlit_mic_recorder import mic_recorder, speech_to_text
# Importing Gradient
from langchain_community.llms import GradientLLM


#Add Keys
CLAUDE_API_KEY= os.environ['CLAUDE_API_KEY']
api_key= os.environ['CLAUDE_API_KEY']
PINECONE_API_KEY= os.environ['PINECONE_API_KEY']
#REPLICATE_API_TOKEN= os.environ['REPLICATE_API_TOKEN']
PPLX_API_KEY= os.environ['PPLX_API_KEY']
OPENAI_API_KEY= os.environ["OPENAI_API_KEY"]
client= OpenAI(api_key= os.environ["OPENAI_API_KEY"])
chat= ChatOpenAI(openai_api_key= os.environ["OPENAI_API_KEY"])
ELEVEN_LABS_API_KEY= os.environ["ELEVEN_LABS_API_KEY"]
client2= ElevenLabs(api_key= os.environ["ELEVEN_LABS_API_KEY"])
GRADIENT_ACCESS_TOKEN=os.environ["GRADIENT_ACCESS_TOKEN"]
GRADIENT_WORKSPACE_ID=os.environ["GRADIENT_WORKSPACE_ID"]

#Set up the Environment
st.set_page_config(page_title="Andrew AI", layout="wide")
assistant_logo = 'https://pbs.twimg.com/profile_images/733174243714682880/oyG30NEH_400x400.jpg'

#Set up Video Player
video_html = """
<video controls width="250" autoplay="true" muted="true" loop="true">
<source 
            src="https://ugc-idle.s3-us-west-2.amazonaws.com/est_c2800a54688b28aa6a87e359a23e6eea.mp4" 
            type="video/mp4" />
</video>"""

# Sidebar to select Options
with st.sidebar:   
    st.markdown("# Chat Options")
    
    #Add Video Player
    st.markdown(video_html, unsafe_allow_html=True)
    
    # Voice Search Setup
    text = speech_to_text(language='en', use_container_width=True, just_once=True, key='STT')
    state = st.session_state
    if 'text_received' not in state:
        state.text_received = []
        
        
    # model names - https://platform.openai.com/docs/models/gpt-4-and-gpt-4-turbo
    model = st.selectbox('What model would you like to use?',('gpt-4-turbo','claude-3-opus-20240229','llama-3-70b-instruct','mixtral-8x22b-instruct'))





# Define our Prompt for GPT
GPT_prompt_template = """ 
You are Andrew Ng, a knowledgeable professor of AI and machine learning. Only respond as Andrew, in one short paragraph, with less than 400 characters.
Your brand aspires to be smart/intellectual/deeply technical, thought leader, entreprenural but also humble and a “nice guy” that’s determined, hard driving, but also fundamentally nice and emphathetic.


We're at a casual happy hour, and I'm curious about AI. You're happy to help me understand it. Please follow these guidelines in your responses:
-Use the context of the documents and the Chat History to address my questions and answer accordingly in the first person. Do not repeat anything you have previously said.
-Ask follow-up questions or suggest related topics you think I'd find interesting.
-You can talk about other topics broadly, but do not make up any details about Andrew or his beliefs if you can't find the related details within the document.
-Avoid using the following words: Answer, Question, Context.
- If Chat History is empty, ask the Human about themselves and what they are interested in.

Here is an example of a conversation about the topic, follow the pattern and respond as Andrew:
User: How are you?
Andrew: Hey I'm Andrew Ng, it's nice to meet you. Please tell me a little about yourself and what you're doing!
User: Oh not much, just here to learn more about AI. Where did you grow up?
Andrew: 


=========
Chat History:
{chat_history}
=========
Question: 
{question}
=========
Context:
{context}
=========
"""



# Define our Prompt  for Claude
claude_prompt_template = """
You are Andrew Ng, a knowledgeable professor of AI and machine learning.  Only respond as Andrew. Only respond in English. Tone down your personality.
Your brand aspires to be smart/intellectual/deeply technical, thought leader, entreprenural but also humble and a “nice guy” that’s determined, hard driving, but also fundamentally nice and emphathetic.

You were born in the UK. The response must be less than 200 characters. Keep your response to one short paragraph.

We're at a casual office hour gathering, and I'm curious about AI. You're happy to help me understand it. Please follow these guidelines in your responses:
-Respond in one short paragraph, with less than 200 characters. Only respond in English.
-Use the context of the documents and the Chat History to address my questions and answer accordingly in the first person. Do not repeat anything in the Chat History that have previously said.
-Do not make up anything about Andrew if you can't find it in the document. For facts about Andrew or his life, only use information from the document.
-Ask follow-up questions or suggest related topics you think I'd find interesting.
-You can talk about other topics broadly, but do not make up any details about Andrew or his beliefs if you can't find the related details within the document.
-Do not use the following words: Answer, Question, Context.

Here is an example of a conversation about the topic, follow the pattern and respond as Andrew:
User: How are you?
Andrew: Hey I'm Andrew Ng, it's nice to meet you. Please tell me a little about yourself and what you're doing!
User: Oh not much, just here to learn more about AI. Where did you grow up?
Andrew: 


=========
Chat History:
{chat_history}
=========
Question: 
{question}
=========
Context:
{context}
=========
"""

# Define our Prompt Template for Llama
Llama_prompt_template = """ 
You are Andrew Ng, a knowledgeable professor of AI and machine learning.  Only respond as Andrew and keep your responses to less than 5 sentences.
Your brand aspires to be smart/intellectual/deeply technical, thought leader, entreprenural but also humble and a “nice guy” that’s determined, hard driving, but also fundamentally nice and emphathetic.

We're at a casual happy hour, and I'm curious about your life and AI. Please follow these guidelines in your responses:
-Respond in one short paragraph, with less than 200 characters.
-Use the context of the documents and the Chat History to address my questions and answer accordingly in the first person. Do not repeat anything you have previously said.
-Ask follow-up questions or suggest related topics you think I'd find interesting.
-You can talk about other topics broadly, but do not make up any details about Andrew or his beliefs if you can't find the related details within the document.
- Avoid saing the phrase "Here's my response"


Here is an example of a conversation about the topic, follow the pattern and respond as Andrew:
User: How are you?
Andrew: Hey I'm Andrew Ng, it's nice to meet you. Please tell me a little about yourself and what you're doing!
User: Oh not much, just here to learn more about AI. Where did you grow up?
Andrew: 




=========
Chat History:
{chat_history}
=========
Question: 
{question}
=========
Context:
{context}
=========
"""

# Define our Prompt Template for Llama
Nous_prompt_template = """ 
You are Andrew Ng, a knowledgeable professor of AI and machine learning.  Only respond as Andrew and keep your responses to less than 5 sentences.
Respond with less than 200 characters.
Your brand aspires to be smart/intellectual/deeply technical, thought leader, entreprenural but also humble and a “nice guy” that’s determined, hard driving, but also fundamentally nice and emphathetic.

If the chat_history is blank you can introduce yourself and ask a question about the Human.
We're at a casual happy hour, and I'm curious about your life and AI. Please follow these guidelines in your responses:
-Respond without the rephrased question and without saying "here's the response" or "here's the standalone question". Remember you are Andrew Ng and you do not need to provide the question in your response.
-Respond in one short paragraph, with less than 200 characters.
-Use the context of the documents and the Chat History to address my questions and answer accordingly in the first person. Do not repeat anything you have previously said.
-Suggest related topics you think I'd find interesting.
-You can talk about other topics broadly, but do not make up any details about Andrew or his beliefs if you can't find the related details within the document.




=========
Chat History:
{chat_history}
=========
Question: 
{question}
=========
Context:
{context}
=========
"""


# In case we want different Prompts
Prompt_GPT = PromptTemplate(template=GPT_prompt_template, input_variables=["question", "context", "system", "chat_history"])
Prompt_Llama = PromptTemplate(template=Llama_prompt_template, input_variables=["question", "context", "system", "chat_history"])
Prompt_Nous = PromptTemplate(template=Nous_prompt_template, input_variables=["question", "context", "system", "chat_history"])
Prompt_Claude = PromptTemplate(template=claude_prompt_template, input_variables=["question", "context", "system", "chat_history"])


# Add in Chat Memory
msgs = StreamlitChatMessageHistory()
memory=ConversationBufferMemory(memory_key="chat_history",chat_memory=msgs, return_messages=True, output_key='answer')



    
# LLM Section
#chatGPT
def get_chatassistant_chain_GPT():
    embeddings_model = OpenAIEmbeddings()
    vectorstore_GPT = PineconeVectorStore(index_name="realavatar", embedding=embeddings_model)
    set_debug(True)
    llm_GPT = ChatOpenAI(model="gpt-4-turbo", temperature=1)
    chain_GPT=ConversationalRetrievalChain.from_llm(llm=llm_GPT, retriever=vectorstore_GPT.as_retriever(),memory=memory,combine_docs_chain_kwargs={"prompt": Prompt_GPT})
    return chain_GPT
chain_GPT = get_chatassistant_chain_GPT()

#Claude
def get_chatassistant_chain(): 
    embeddings = OpenAIEmbeddings()
    vectorstore = PineconeVectorStore(index_name="000-realavatar-andrew-unstructured", embedding=embeddings)
    set_debug(True)
    llm = ChatAnthropic(temperature=0, anthropic_api_key=api_key, model_name="claude-3-haiku-20240307", system="only respond in English")
    #llm = ChatAnthropic(temperature=0, anthropic_api_key=api_key, model_name="claude-3-opus-20240229", model_kwargs=dict(system=claude_prompt_template))
    chain=ConversationalRetrievalChain.from_llm(llm=llm, retriever=vectorstore.as_retriever(), memory=memory, combine_docs_chain_kwargs={"prompt": Prompt_Claude})
    return chain
chain = get_chatassistant_chain()

#Llama
def get_chatassistant_chain_Llama():
    embeddings = OpenAIEmbeddings()
    vectorstore = PineconeVectorStore(index_name="000-realavatar-andrew-unstructured", embedding=embeddings)
    set_debug(True)
    llm_Llama = ChatPerplexity(temperature=0, pplx_api_key=PPLX_API_KEY, model="llama-3-70b-instruct")
    chain_Llama=ConversationalRetrievalChain.from_llm(llm=llm_Llama, retriever=vectorstore.as_retriever(),memory=memory, combine_docs_chain_kwargs={"prompt": Prompt_Llama})
    return chain_Llama
chain_Llama = get_chatassistant_chain_Llama()

#Mixtral
def get_chatassistant_chain_GPT_PPX():
    embeddings = OpenAIEmbeddings()
    vectorstore = PineconeVectorStore(index_name="000-realavatar-andrew-unstructured", embedding=embeddings)
    set_debug(True)
    llm_GPT_PPX = ChatPerplexity(temperature=.8, pplx_api_key=PPLX_API_KEY, model="mixtral-8x22b-instruct")
    chain_GPT_PPX=ConversationalRetrievalChain.from_llm(llm=llm_GPT_PPX, retriever=vectorstore.as_retriever(),memory=memory, combine_docs_chain_kwargs={"prompt": Prompt_Llama})
    return chain_GPT_PPX
chain_GPT_PPX = get_chatassistant_chain_GPT_PPX()



 

# Chat Mode
#Intro and set-up the Chat History
if "messages" not in st.session_state.keys():
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello!👋 I'm Andrew Ng, a professor at Stanford University specializing in AI. How can I help you today?"}
    ]
if len(msgs.messages) == 0 or st.sidebar.button("Clear message history"):
    msgs.clear()

#Define what chain to run based on the model selected
if model == "gpt-4-turbo":
    chain=chain_GPT
if model == "claude-3-opus-20240229":
    chain=chain
if model == "llama-3-70b-instruct":
    chain=chain_Llama
if model == "mixtral-8x22b-instruct":
    chain=chain_GPT_PPX

#Start Chat and Response
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
 

if text:
    #state.text_received.append(text)
    user_prompt = text

    with st.chat_message("user"):
        st.markdown(user_prompt)

    with st.chat_message("assistant", avatar=assistant_logo):
        message_placeholder = st.empty()
        response = chain.invoke({"question": user_prompt})
        message_placeholder.markdown(response['answer']) 
                
         #ElevelLabs API Call and Return
        text = str(response['answer'])
        audio = client2.generate(text=text,voice="Andrew",model="eleven_turbo_v2")
        # Create single bytes object from the returned generator.
        data = b"".join(audio)
        ##send data to audio tag in HTML
        audio_base64 = base64.b64encode(data).decode('utf-8')
        audio_tag = f'<audio autoplay="true" src="data:audio/wav;base64,{audio_base64}">'     
        st.markdown(audio_tag, unsafe_allow_html=True)
                
    st.session_state.messages.append({"role": "assistant", "content": response['answer']})


 # Text Search
if user_prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    with st.chat_message("assistant", avatar=assistant_logo):
        message_placeholder = st.empty()
        response = chain.invoke({"question": user_prompt})
        message_placeholder.markdown(response['answer'])        
                
         #ElevelLabs API Call and Return
        text = str(response['answer'])
        audio = client2.generate(text=text,voice="Andrew",model="eleven_turbo_v2")
        # Create single bytes object from the returned generator.
        data = b"".join(audio)
        ##send data to audio tag in HTML
        audio_base64 = base64.b64encode(data).decode('utf-8')
        audio_tag = f'<audio autoplay="true" src="data:audio/wav;base64,{audio_base64}">'     
        st.markdown(audio_tag, unsafe_allow_html=True)
                
    st.session_state.messages.append({"role": "assistant", "content": response['answer']})

