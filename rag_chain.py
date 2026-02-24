import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_classic.chains import RetrievalQA
from langchain_classic.prompts import PromptTemplate
from vector_store import get_vector_store

# Load environment variables
load_dotenv()

def setup_rag_chain():
    """
    Sets up the RAG chain using classic RetrievalQA.
    """
    # Initialize the LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-flash-latest",
        temperature=0.2, # Lower temperature for more factual responses
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )

    # Load the vector store with local embeddings
    vector_store = get_vector_store()
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})

    # Define the prompt template
    template = """
    You are a professional medical assistant based on the Uganda Clinical Guidelines and WHO standards.
    Use the following pieces of retrieved context to answer the user's question. 
    If you don't know the answer based on the context, say that you don't know, but provide general guidance 
    to consult a health professional.
    
    Always include this disclaimer in your response: 
    "DISCLAIMER: This information is for educational purposes based on clinical guidelines. 
    It does not replace professional medical advice. For emergencies, please visit a health facility immediately."

    Context:
    {context}

    Question: 
    {question}

    Helpful Answer:
    """
    
    prompt = PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )

    # Create the RAG chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )
    
    return qa_chain

def ask_health_question(question: str):
    """
    Main function to ask a health question and get a RAG-based response.
    """
    chain = setup_rag_chain()
    # RetrievalQA uses "query" as input key
    response = chain.invoke({"query": question})
    
    return {
        "answer": response["result"],
        "sources": [doc.metadata.get("source") for doc in response["source_documents"]]
    }

if __name__ == "__main__":
    # Test session
    print("Testing RAG chain...")
    test_question = "How do I treat malaria according to the guidelines?"
    try:
        result = ask_health_question(test_question)
        print("\nAnswer:\n", result["answer"])
        print("\nSources:\n")
        for source in set(result["sources"]):
            print(f"- {source}")
    except Exception as e:
        print(f"Error: {e}")
