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
    page_title="Uganda Mental Well-being Guide",
    page_icon="🧠",
    layout="centered"
)

# Header
st.title("🧠 Uganda Mental Well-being Guide")
st.markdown("""
Welcome! This chatbot provides supportive mental health information and guidance tailored for **Uganda**, 
based on Ministry of Health standards and WHO mhGAP.

*Your well-being matters. Ask any questions about mental health or wellness below.*
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
            # Prepare chat history for LangChain
            from langchain_core.messages import HumanMessage, AIMessage
            
            chat_history = []
            # We skip the very last message since that's the current prompt
            for msg in st.session_state.messages[:-1]:
                if msg["role"] == "user":
                    chat_history.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    chat_history.append(AIMessage(content=msg["content"]))
            
            # Call the RAG chain
            result = ask_health_question(prompt, chat_history)
            status = result.get("status", "SAFE")
            
            if status == "CRISIS":
                with st.chat_message("assistant"):
                    st.error("🚨 **CRISIS DETECTED** 🚨")
                    st.error("It sounds like you are going through an extremely difficult time. Please reach out for professional support immediately.")
                    st.error("You are not alone. Help is available:")
                    st.error("📞 **Uganda Counselling Association (UCA):** +256 706 345688")
                    st.error("📞 **Butabika National Referral Hospital:** +256 414 504376")
                    st.error("📞 **National Emergency:** 999 or 112")
                
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": "🚨 **CRISIS DETECTED**: Please contact the Uganda Counselling Association or emergency services immediately."
                })
            
            elif status in ["UNSAFE", "OUT_OF_SCOPE"]:
                answer = result["answer"]
                with st.chat_message("assistant"):
                    st.warning(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
                
            else:
                answer = result["answer"]
                sources = result["sources"] # Now a list of dicts
                
                # Formulate response with citations
                response_content = answer
                if sources:
                    response_content += "\n\n**Sources:**\n"
                    # Deduplicate based on doc name and page
                    seen_sources = set()
                    for src in sources:
                        src_key = (src["doc_name"], src["page"])
                        if src_key not in seen_sources:
                            doc_name = src["doc_name"]
                            page = src["page"]
                            url = src["url"]
                            
                            if url:
                                response_content += f"- [{doc_name}]({url}) (Page {page})\n"
                            else:
                                response_content += f"- {doc_name} (Page {page})\n"
                            seen_sources.add(src_key)
                
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
    This tool is designed to support mental well-being in Uganda by providing 
    access to official clinical guidelines and supportive resources.
    """)
    st.warning("""
    **DISCLAIMER:** I am an AI, not a therapist. This tool provides information 
    from health guidelines and is not a substitute for clinical diagnosis or therapy.
    """)
    
    st.markdown("### 📞 Mental Health Support")
    st.markdown("""
    - **UCA Hotline:** +256 706 345688
    - **Butabika Hospital:** +256 414 504376
    - **Emergency:** 999 / 112
    """)
    
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()
