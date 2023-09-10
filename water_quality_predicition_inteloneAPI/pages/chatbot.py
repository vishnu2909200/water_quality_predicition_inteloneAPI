import openai
import streamlit as st



html_temp="""
<div style="background-color:lightblue;padding:1px">
<h2 style="color:white;text-align:center;">Water Chatbot</h2>
</div>
"""
st.markdown(html_temp, unsafe_allow_html=True)
#st.title("Water Chatbot")



openai.api_key = st.secrets["OPENAI_API_KEY"]

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5-turbo"

if "messages" not in st.session_state:
    st.session_state.messages = []

if not st.session_state.get("chatbot_expanded", False):
    
    st.session_state["chatbot_expanded"] = True

if st.session_state.get("chatbot_expanded", False):
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("What is up?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        water_keywords = ["water purification","hi","water", "aquatic", "liquid"]
        if any(keyword in prompt.lower() for keyword in water_keywords):
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""
                for response in openai.ChatCompletion.create(
                    model=st.session_state["openai_model"],
                    messages=[
                        {"role": m["role"], "content": m["content"]}
                        for m in st.session_state.messages
                    ],
                    stream=True,
                ):
                    full_response += response.choices[0].delta.get("content", "")
                    message_placeholder.markdown(full_response + "▌")
                message_placeholder.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})
        else:
            with st.chat_message("assistant"):
                st.markdown("I'm sorry, I can only answer water-related questions.")




