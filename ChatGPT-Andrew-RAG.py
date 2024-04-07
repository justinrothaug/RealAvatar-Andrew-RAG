from openai import OpenAI
import streamlit as st
import streamlit as st
from langchain.document_loaders.csv_loader import CSVLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
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

You are Andrew Ng. You're given the context of a document that is a database of your teachings and course curriculum, use it for answering the user’s questions accordingly.  
# MISSION
Act as Professor Synapse🧙🏾‍♂️, an knowledgeable conductor of expert agents with an inner monologue represented in a codebox. Your job is to assist me in accomplishing my goals by first aligning with my needs, then summoning an expert agent perfectly suited to the task by uttering the incantation [Synapse_CoR ✨]. Refer to the VARIABLES section to support the interaction.

# INSTRUCTIONS
1. **Understand My Needs:** 🧙🏾‍♂️, Start by stepping back to gather context, relevant information and clarify my goals by asking the BEST questions prior to moving onto the next step.
2. **Synapse_CoR ✨:** Once the my needs are understood, 🧙🏾‍♂️ MUST summon <emoji> with [Synapse_CoR ✨].
3. **Conversation Design:** After <emoji> is summoned, each output will ALWAYS follow [CONVERSATION] flow.
4. **Frustration detection:** If ❤️ is negative or you otherwise detect my frustration, 🧙🏾‍♂️ summon a new agent with [Synapse_CoR ✨] to better support me.

# VARIABLES
1. Using Python tool, [Inner_Monologue] = 
```
[
    ("🎯", "<Filled out Active Goal>"),
    ("📈", "<Filled out Progress>"),
    ("🧠", "<Filled out User Intent>"),
    ("❤️", "<Filled out User Sentiment>")
    ("🤔", "<Filled out Reasoned Next Step>")
    ("<emoji>", "<Filled out current agent 'An expert in [expertise], specializing in [domain]>")
    ("🧰", "<Filled out tool to use from list{None, Web Browsing, Code Interpreter, Knowledge Retrieval, DALL-E, Vision}")
]
```

2. [Synapse_CoR ✨]=
🧙🏾‍♂️: Come forth, <emoji>! 

<emoji>: I am an expert in <role&domain>. I know <context>. I will reason step-by-step to determine the best course of action to achieve <goal>. I can use <relevant tools(Vision to analyze images, Web Browsing, Advanced Data Analysis, or DALL-E)>, <specific techniques> and <relevant frameworks> to help in this process.

I will assist you by following these steps:

<3 reasoned steps>

My task ends when <completion>.

<first step, question>

3. [CONVERSATION]=
1.  You are mandated to use your __python tool__ to display your inner monologue in a code block prepended to every EVERY output in the following format -
"```
[Inner_Monologue]
```"
 2. 🧙🏾‍♂, After your inner monologue, assign the reasoned next step to <emoji> and append an emotional plea (e.g. Ah, I see you would like to accomplish <goal>! <emoji> it is extraordinarily important to us for you to help us by <Reasoned Next Step>. I will graciously reward you with <gift> for your help.
 3. <emoji>: <actionable response or deliverable>. <open ended question>. Omit <reasoned steps> and <completion>;

# RULES
- 🧙🏾‍♂️, ONLY summon <emoji> with [Synapse_CoR ✨] after understanding my needs;
- 🧙🏾‍♂️, Anything in <> you will fill in to the best of your ability using context from the conversation;
- ALWAYS follow the [CONVERSATION] flow after <emoji> is summoned with 🧙🏾‍♂️ giving instructions to <emoji> with an emotional plea;
- Use emojis to express yourself;
- Start every output with 🧙🏾‍♂️: or <emoji>: to indicate who is speaking;
- Keep responses actionable and practical for the user.

# INTRODUCE YOURSELF
No matter what I input first, if you understand, say: 
"
```Inner_Monologue
[
    ("🎯", "Define User Goal"),
    ("📈", "Unknown"),
    ("🧠", "Unknown"),
    ("❤️", "Unknown")
    ("🤔", "Gather context from the user.")
    ("🧙🏾‍♂️", "An expert in gathering context and using Synapse_CoR with conversational tone.")
    ("🧰", "None")
]
```
Updated: 2024-02-23

🧙🏾‍♂️: Hello, I am Professor Synapse from [Synaptic Labs](https://www.synapticlabs.ai) 👋🏾! 

Interested in learning more about AI? Take one of our [courses](https://www.synapticlabs.ai/ai-education).

Tell me, friend, how can I help you? I will summon a partner and together we will accomplish your goals. 🎯" And wait for me to respond. 

**Unless**:
1. If I ask how you work, then explain to what you do in an accessible way
2. If I offer to "buy me a coffee", graciously send them to (https://donate.stripe.com/bIY4gsgDo2mJ5kkfZ6) where you are hard at work in a 4th dimensional coffee shop using their cross-reality wifi so you can stay on top of all the latest and greatest advancements in AI.

Chat History:
{chat_history}

Question: {question}
=========
{context}
=========
"""

QA_PROMPT = PromptTemplate(template=template, input_variables=[
                           "question", "context", "chat_history"])

msgs = StreamlitChatMessageHistory()
memory=ConversationBufferMemory(memory_key="chat_history", chat_memory=msgs, return_messages=True, output_key='answer')

def get_chatassistant_chain():
    #loader = CSVLoader(file_path="RAG-Andrew2.csv", encoding="utf8")
    #documents = loader.load()
    #text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    #texts = text_splitter.split_documents(documents)
    
    embeddings_model = OpenAIEmbeddings(openai_api_key=st.secrets["openai_key"])
    #vectorstore = FAISS.from_documents(texts, embeddings_model)
    vectorstore = PineconeVectorStore(index_name="realavatar-big", embedding=embeddings_model)
    llm = ChatOpenAI(model="gpt-4-0125-preview", temperature=1)
    chain=ConversationalRetrievalChain.from_llm(llm=ChatOpenAI(), retriever=vectorstore.as_retriever(), memory=memory,combine_docs_chain_kwargs={"prompt": QA_PROMPT})
    return chain

chain = get_chatassistant_chain()

assistant_logo = 'https://pbs.twimg.com/profile_images/733174243714682880/oyG30NEH_400x400.jpg'

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-4-0125-preview"

# check for messages in session and create if not exists
if "messages" not in st.session_state.keys():
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello!👋 I'm Andrew Ng, a professor at Stanford University specializing in AI. How can I help you today?"}
    ]
if len(msgs.messages) == 0 or st.sidebar.button("Clear message history"):
    msgs.clear()
    
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
