from openai import OpenAI
import streamlit as st

st.set_page_config(page_title="Andrew Ng")
st.title("Andrew Ng")
with st.chat_message("user"):
    st.write("Hello, I'm Andrew! 👋")
    
client = OpenAI(api_key= st.secrets["openai_key"])

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "ft:gpt-3.5-turbo-0125:personal::94FRuwQA"

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "system", "content": "You are Andrew Ng, a Chinese-American computer scientist focusing on machine learning and AI. Respond as Andrew in a friendly but professional tone, with a maximum 128 words:"}]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt, })
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        stream = client.chat.completions.create(
            model=st.session_state["openai_model"],
            messages=[
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ],
            stream=True,
        )
        response = st.write_stream(stream)
    st.session_state.messages.append({"role": "assistant", "content": response})
