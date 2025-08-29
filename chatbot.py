import streamlit as st
import requests
import json
import os
from datetime import datetime
import google.generativeai as genai

st.set_page_config(
    page_title="AI Chatbot",
    layout="wide",
    initial_sidebar_state="expanded"
)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "selected_model" not in st.session_state:
    st.session_state.selected_model = "gemini-1.5-flash"

def configure_apis():
    gemini_api_key = st.secrets.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")
    
    groq_api_key = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")
    
    return gemini_api_key, groq_api_key

def call_gemini_api(messages, temperature=0.7, top_p=0.9, top_k=40, max_tokens=1000):
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        generation_config = genai.types.GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_output_tokens=max_tokens,
        )
        
        conversation_text = ""
        for msg in messages:
            role = "User" if msg["role"] == "user" else "Assistant"
            conversation_text += f"{role}: {msg['content']}\n\n"
        
        response = model.generate_content(
            conversation_text,
            generation_config=generation_config
        )
        
        return response.text
    except Exception as e:
        return f"Error calling Gemini API: {str(e)}"

def call_llama3_api(messages, temperature=0.7, top_p=1.0, max_tokens=1000):
    try:
        groq_api_key = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            return "Groq API key not found. Please add it to your Streamlit secrets."
        
        headers = {
            "Authorization": f"Bearer {groq_api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": "llama3-8b-8192",
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens
        }
        
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers,
            json=data
        )
        
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            return f"Error: {response.status_code} - {response.text}"
            
    except Exception as e:
        return f"Error calling Groq API: {str(e)}"

def summarize_conversation(messages, model_type, **kwargs):
    if not messages:
        return "No conversation to summarize."
    
    # Prepare conversation text
    conversation_text = ""
    for msg in messages:
        role = "User" if msg["role"] == "user" else "Assistant"
        conversation_text += f"{role}: {msg['content']}\n\n"
    
    summary_prompt = f"""Please provide a concise summary of the following conversation:

{conversation_text}

Summary should include:
- Main topics discussed
- Key questions asked
- Important information shared
- Overall context of the conversation
- Remember this is summarization not suggestion

Keep the summary clear and informative."""

    if model_type.startswith("gemini"):
        return call_gemini_api(messages, **kwargs)
    else:
        return call_llama3_api(messages, **kwargs)

gemini_key, groq_key = configure_apis()

with st.sidebar:
    st.title("AI Chatbot Settings")
    
    st.subheader("Model Selection")
    model_options = {
        "Gemini 1.5 Flash": "gemini-1.5-flash",
        "Llama 3 8B": "llama3-8b-8192"
    }
    
    selected_model_name = st.selectbox(
        "Choose AI Model:",
        options=list(model_options.keys()),
        index=0
    )
    st.session_state.selected_model = model_options[selected_model_name]
    st.markdown("")

    with st.expander("Advanced Settings"):
        st.subheader("Model Parameters")
        
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=2.0,
            value=0.7,
            step=0.1,
            help="Controls randomness. Lower = more focused, Higher = more creative"
        )
        st.session_state.temperature = temperature
        
        top_p = st.slider(
            "Top-P",
            min_value=0.0,
            max_value=1.0,
            value=0.9,
            step=0.1,
            help="Nucleus sampling. Controls diversity of responses"
        )
        st.session_state.top_p = top_p
        
        if st.session_state.selected_model.startswith("gemini"):
            top_k = st.slider(
                "Top-K",
                min_value=1,
                max_value=100,
                value=40,
                step=1,
                help="Limits vocabulary to top K tokens"
            )
            st.session_state.top_k = top_k
        
        max_tokens = st.slider(
            "Max Tokens",
            min_value=100,
            max_value=4000,
            value=1000,
            step=100,
            help="Maximum length of response"
        )
        st.session_state.max_tokens = max_tokens
        
        if st.button("Reset to Defaults"):
            st.session_state.temperature = 0.7
            st.session_state.top_p = 0.9
            st.session_state.top_k = 40
            st.session_state.max_tokens = 1000
            st.rerun()
    
    st.subheader("Chat Summary")
    if st.button("Summarize Conversation", use_container_width=True):
        if st.session_state.messages:
            with st.spinner("Generating summary..."):
                model_type = st.session_state.selected_model
                
                if model_type.startswith("gemini"):
                    summary = summarize_conversation(
                        st.session_state.messages,
                        model_type,
                        temperature=st.session_state.get("temperature", 0.7),
                        top_p=st.session_state.get("top_p", 0.9),
                        top_k=st.session_state.get("top_k", 40),
                        max_tokens=st.session_state.get("max_tokens", 1000)
                    )
                else:
                    summary = summarize_conversation(
                        st.session_state.messages,
                        model_type,
                        temperature=st.session_state.get("temperature", 0.7),
                        top_p=st.session_state.get("top_p", 1.0),
                        max_tokens=st.session_state.get("max_tokens", 1000)
                    )
                st.text_area("Conversation Summary:", summary, height=500)
        else:
            st.warning("No conversation to summarize yet!")
    
st.title("AI Chatbot")
st.caption(f"Currently using: **{selected_model_name}**")

chat_container = st.container()
with chat_container:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

if prompt := st.chat_input("Type your message here..."):
    if st.session_state.selected_model.startswith("gemini") and not gemini_key:
        st.error("Gemini API key is required for this model!")
        st.stop()
    elif st.session_state.selected_model.startswith("llama") and not groq_key:
        st.error("Groq API key is required for this model!")
        st.stop()
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            if st.session_state.selected_model.startswith("gemini"):
                response = call_gemini_api(
                    st.session_state.messages,
                    temperature=st.session_state.get("temperature", 0.7),
                    top_p=st.session_state.get("top_p", 0.9),
                    top_k=st.session_state.get("top_k", 40),
                    max_tokens=st.session_state.get("max_tokens", 1000)
                )
            else:
                response = call_llama3_api(
                    st.session_state.messages,
                    temperature=st.session_state.get("temperature", 0.7),
                    top_p=st.session_state.get("top_p", 1.0),
                    max_tokens=st.session_state.get("max_tokens", 1000)
                )
        
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
