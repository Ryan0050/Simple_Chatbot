import streamlit as st
import requests
import os
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
if "bot_behavior" not in st.session_state:
    st.session_state.bot_behavior = "Default"
if "bot_persona" not in st.session_state:
    st.session_state.bot_persona = ""

def configure_apis():
    gemini_api_key = st.secrets.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")
    if gemini_api_key:
        genai.configure(api_key=gemini_api_key)
    
    groq_api_key = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")
    
    return gemini_api_key, groq_api_key

def construct_system_prompt():
    behavior = st.session_state.get("bot_behavior", "Default")
    persona = st.session_state.get("bot_persona", "").strip()
    
    prompts = []
    if behavior != "Default":
        prompts.append(f"Your response style must be: {behavior.lower()}.")
    
    if persona:
        prompts.append(f"You must adhere to the following persona or context: {persona}")
        
    return " ".join(prompts)

def call_gemini_api(messages, system_prompt="", temperature=0.7, top_p=0.9, top_k=40, max_tokens=1000):
    try:
        model = genai.GenerativeModel(
            'gemini-1.5-flash',
            system_instruction=system_prompt
        )
        
        generation_config = genai.types.GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_output_tokens=max_tokens,
        )
        
        gemini_history = []
        for msg in messages:
            role = "model" if msg["role"] == "assistant" else "user"
            gemini_history.append({"role": role, "parts": [msg["content"]]})
        
        response = model.generate_content(
            gemini_history,
            generation_config=generation_config
        )
        
        return response.text
    except Exception as e:
        if "api_key" in str(e).lower():
            return "Error: Gemini API key is invalid or not set."
        return f"Error calling Gemini API: {str(e)}"

def call_llama3_api(messages_with_system, temperature=0.7, top_p=1.0, max_tokens=1000):
    try:
        groq_api_key = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            return "Groq API key not found. Please add it to your Streamlit secrets."
        
        headers = {
            "Authorization": f"Bearer {groq_api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": "llama-3.1-8b-instant",
            "messages": messages_with_system,
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

    summary_messages = [{"role": "user", "content": summary_prompt}]

    if model_type.startswith("gemini"):
        return call_gemini_api(summary_messages, system_prompt="", **kwargs)
    else:
        return call_llama3_api(summary_messages, **kwargs)

gemini_key, groq_key = configure_apis()

with st.sidebar:
    st.title("AI Chatbot Settings")
    
    st.subheader("Model Selection")
    model_options = {
        "Gemini 1.5 Flash": "gemini-1.5-flash",
        "Llama 3.1 8B": "llama-3.1-8b-instant"
    }
    
    selected_model_name = st.selectbox(
        "Choose AI Model:",
        options=list(model_options.keys()),
        index=0
    )
    st.session_state.selected_model = model_options[selected_model_name]
    st.markdown("")

    st.subheader("Bot Personality")
    
    behavior_options = ["Default", "Professional", "Enthusiastic", "Humorous", "Sarcastic", "Brief and to the point"]
    st.session_state.bot_behavior = st.selectbox(
        "Select Bot Behavior:",
        options=behavior_options,
        key="bot_behavior_select"
    )
    
    st.session_state.bot_persona = st.text_area(
        "Custom Persona / Context:",
        placeholder="e.g., You are a senior software engineer...",
        height=100,
        key="bot_persona_text"
    )
    st.markdown("")

    with st.expander("Advanced Settings"):
        st.subheader("Model Parameters")
        
        temperature = st.slider(
            "Temperature", 0.0, 2.0, 0.7, 0.1,
            help="Controls randomness. Lower = more focused, Higher = more creative"
        )
        st.session_state.temperature = temperature
        
        top_p = st.slider(
            "Top-P", 0.0, 1.0, 0.9, 0.1,
            help="Nucleus sampling. Controls diversity of responses"
        )
        st.session_state.top_p = top_p
        
        if st.session_state.selected_model.startswith("gemini"):
            top_k = st.slider(
                "Top-K", 1, 100, 40, 1,
                help="Limits vocabulary to top K tokens"
            )
            st.session_state.top_k = top_k
        
        max_tokens = st.slider(
            "Max Tokens", 100, 4000, 1000, 100,
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
            
            system_prompt = construct_system_prompt()
            
            if st.session_state.selected_model.startswith("gemini"):
                response = call_gemini_api(
                    st.session_state.messages,
                    system_prompt=system_prompt,
                    temperature=st.session_state.get("temperature", 0.7),
                    top_p=st.session_state.get("top_p", 0.9),
                    top_k=st.session_state.get("top_k", 40),
                    max_tokens=st.session_state.get("max_tokens", 1000)
                )
            else:
                messages_with_system = []
                if system_prompt:
                    messages_with_system.append({"role": "system", "content": system_prompt})
                messages_with_system.extend(st.session_state.messages)
                
                response = call_llama3_api(
                    messages_with_system,
                    temperature=st.session_state.get("temperature", 0.7),
                    top_p=st.session_state.get("top_p", 1.0),
                    max_tokens=st.session_state.get("max_tokens", 1000)
                )
        
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
