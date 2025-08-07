import os
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
#step 1 SETUP L
load_dotenv()
GROQ_API_KEY=os.getenv("GROQ_API_KEY")


def load_llm():
    llm= ChatGroq(
        model_name="meta-llama/llama-4-maverick-17b-128e-instruct",
        temperature=0.0,
        groq_api_key=GROQ_API_KEY
    )
    return llm


#step 2 CONNECT LLM with faiss
DB_FAISS_PATH="vectorstore/db_faiss"

custom_prompt_template="""use the pieces of information provided in the context to answer the user question. If you dont know the anasawr jsut say that you dont know dont try to make up an answer.Dont provide anything out of the given context

Context:{context}
Question:{question}

Start answer directly no small talk please.
"""

def set_custom_prompt(custom_prompt_template):
    prompt=PromptTemplate(template=custom_prompt_template,input_variables=["context","question"])
    return prompt

#load db
embedding_model=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-V2")
db=FAISS.load_local(DB_FAISS_PATH,embedding_model,allow_dangerous_deserialization=True)

#create qa chain
qa_chain=RetrievalQA.from_chain_type(
    llm=load_llm(),
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={'k':3}),
    return_source_documents=True,
    chain_type_kwargs={'prompt':set_custom_prompt(custom_prompt_template)}
)

#use
user_query=input("Write Query here:")
response=qa_chain.invoke({'query':user_query})
print("RESULT", response["result"])
print("SOURCEDOCS", response["source_documents"])

