import streamlit as st
import os
from dotenv import load_dotenv
from rag_chain import ask_health_question

# Load environment variables
load_dotenv()

# Verify GROQ API key
if not os.getenv("GROQ_API_KEY"):
    st.error("GROQ_API_KEY is not set in the .env file. Please add it to enable guardrails.")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="Uganda Health Guide Chatbot",
    page_icon="🏥",
    layout="centered"
)

# Header
st.title("🏥 Uganda Health Guide")
st.markdown("""
Welcome! This chatbot provides medical information based on the **Uganda Clinical Guidelines 2023** 
and WHO standards. 

*Enter your health-related questions below.*
""")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("Ask a health question..."):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.spinner("Searching medical guidelines..."):
        try:
            # Call the RAG chain
            result = ask_health_question(prompt)
            status = result.get("status", "SAFE")
            
            if status == "CRISIS":
                with st.chat_message("assistant"):
                    st.error("🚨 **CRISIS DETECTED** 🚨")
                    st.error("This appears to be a medical emergency or crisis situation.")
                    st.error("Please seek immediate help from a medical professional or contact emergency services:")
                    st.error("📞 **Uganda National Emergency:** 999 or 112")
                    st.error("📞 **Child Helpline:** 116")
                
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": "🚨 **CRISIS DETECTED**: Please contact emergency services (999 or 112) immediately."
                })
            
            elif status in ["UNSAFE", "OUT_OF_SCOPE"]:
                answer = result["answer"]
                with st.chat_message("assistant"):
                    st.warning(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
                
            else:
                answer = result["answer"]
                sources = list(set(result["sources"]))
                
                # Formulate response with citations
                response_content = answer
                if sources:
                    response_content += "\n\n**Sources:**\n"
                    for i, src in enumerate(sources):
                        response_content += f"- {os.path.basename(src)}\n"
                
                # Display assistant response in chat message container
                with st.chat_message("assistant"):
                    st.markdown(response_content)
                
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": response_content})
            
        except Exception as e:
            error_msg = f"Sorry, I encountered an error: {str(e)}"
            st.error(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})

# Sidebar
with st.sidebar:
    st.header("About")
    st.info("""
    This tool is designed to assist health workers and the public in Uganda 
    to quickly find information from official management guidelines.
    """)
    st.warning("""
    **DISCLAIMER:** This is an AI assistant. It does not replace professional medical judgment. 
    Always consult a qualified health professional for diagnosis and treatment.
    """)
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()
