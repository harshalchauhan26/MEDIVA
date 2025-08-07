import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS

DB_FAISS_PATH = "vectorstore/db_faiss"

@st.cache_resource
def get_vs():
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db

def format_sources(sources):
    formatted = ""
    for i, doc in enumerate(sources):
        metadata = doc.metadata
        source = metadata.get('source', 'Unknown Source')
        page = metadata.get('page', 'N/A')
        content = doc.page_content.strip().replace("\n", " ")
        formatted += f"\n**Source {i+1}:** `{source}` (Page {page})\n> {content[:300]}...\n"
    return formatted

def set_custom_prompt(custom_prompt_template):
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt

def load_llm(GROQ_API_KEY):
    llm = ChatGroq(
        model_name="meta-llama/llama-4-maverick-17b-128e-instruct",
        temperature=0.0,
        max_tokens=1000,
        groq_api_key=GROQ_API_KEY
    )
    return llm

def main():
    st.markdown("""
    <style>
    body, .stApp {
        background: linear-gradient(135deg, #0F2027, #203A43, #2C5364);
        color: #E0E0E0;
        font-family: 'Segoe UI', sans-serif;
    }
    .main-title {
        font-size: 48px;
        font-weight: 900;
        text-align: center;
        color: #BB86FC;
        margin-top: 30px;
        animation: fadeInScale 1.2s ease-in-out forwards;
    }
    .sub-text {
        font-size: 20px;
        text-align: center;
        margin-bottom: 30px;
        color: #B0B0B0;
        animation: fadeIn 2s ease-in-out;
    }
    @keyframes fadeIn {
        0% {opacity: 0;}
        100% {opacity: 1;}
    }
    @keyframes fadeInScale {
        0% {opacity: 0; transform: scale(0.95);}
        100% {opacity: 1; transform: scale(1);}
    }
    .chat-bubble-user, .chat-bubble-assistant {
        padding: 16px;
        border-radius: 15px;
        margin-bottom: 12px;
        animation: fadeIn 0.8s ease-in-out;
    }
    .chat-bubble-user {
        background-color: rgba(255, 255, 255, 0.05);
        border-left: 5px solid #03DAC6;
    }
    .chat-bubble-assistant {
        background-color: rgba(255, 255, 255, 0.08);
        border-left: 5px solid #BB86FC;
    }
    .source-docs {
        max-height: 200px;
        overflow-y: auto;
        font-size: 14px;
        background-color: rgba(0, 0, 0, 0.4);
        padding: 12px;
        border-radius: 10px;
        border: 1px solid #444;
        margin-top: 10px;
        color: #B0B0B0;
    }
    .bubble-background {
        position: fixed;
        top: 0;
        left: 0;
        z-index: -1;
        width: 100%;
        height: 100%;
        overflow: hidden;
    }
    .bubble {
        position: absolute;
        bottom: -100px;
        background: rgba(255, 255, 255, 0.05);
        border-radius: 50%;
        animation: rise 40s infinite ease-in;
    }
    @keyframes rise {
        0% {
            transform: translateY(0) scale(0.5);
            opacity: 0.2;
        }
        100% {
            transform: translateY(-1200px) scale(1);
            opacity: 0;
        }
    }
    </style>
    <div class="bubble-background">
        <div class="bubble" style="width: 60px; height: 60px; left: 10%; animation-delay: 0s;"></div>
        <div class="bubble" style="width: 100px; height: 100px; left: 20%; animation-delay: 10s;"></div>
        <div class="bubble" style="width: 40px; height: 40px; left: 30%; animation-delay: 5s;"></div>
        <div class="bubble" style="width: 80px; height: 80px; left: 50%; animation-delay: 3s;"></div>
        <div class="bubble" style="width: 30px; height: 30px; left: 70%; animation-delay: 7s;"></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='main-title'>MEDIVA: Medical RAG Chatbot</div>", unsafe_allow_html=True)
   

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        role_class = "chat-bubble-user" if message['role'] == 'user' else "chat-bubble-assistant"
        st.markdown(f"<div class='{role_class}'>{message['content']}</div>", unsafe_allow_html=True)

    prompt = st.chat_input("Ask your medical question here...")

    if prompt:
        st.markdown(f"<div class='chat-bubble-user'>{prompt}</div>", unsafe_allow_html=True)
        st.session_state.messages.append({'role': 'user', 'content': prompt})

        custom_prompt_template = """use the pieces of information provided in the context to answer the user question. If you dont know the answer just say that you dont know dont try to make up an answer.Dont provide anything out of the given context

         Context:{context}
         Question:{question}

        Start answer directly no small talk please.
          """
        from dotenv import load_dotenv
        load_dotenv()

        GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

        try:
            vectorstore = get_vs()
            if vectorstore is None:
                st.error("FAILED TO LOAD")

            qa_chain = RetrievalQA.from_chain_type(
                llm=ChatGroq(
                    model_name="meta-llama/llama-4-maverick-17b-128e-instruct",
                    temperature=0.0,
                    groq_api_key=os.environ["GROQ_API_KEY"],
                ),
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
                return_source_documents=True,
                chain_type_kwargs={'prompt': set_custom_prompt(custom_prompt_template)}
            )

            response = qa_chain.invoke({'query': prompt})
            result = response["result"]
            sources = format_sources(response["source_documents"])

            result_to_show = result + "\n\n<div class='source-docs'><b>Source Docs:</b>\n" + str(sources) + "</div>"

            st.markdown(f"<div class='chat-bubble-assistant'>{result_to_show}</div>", unsafe_allow_html=True)
            st.session_state.messages.append({'role': 'assistant', 'content': result_to_show})

        except Exception as e:
            st.error(f"ERROR: {str(e)}")

if __name__ == "__main__":
    main()
