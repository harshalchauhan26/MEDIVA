from langchain_community.document_loaders import PyPDFLoader,DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())


#STEP 1 LOADING RAW PDFS TO  GAIN KNOWLEDGE
DATA_PATH="data/"
def load_pdf(data):
    
    loader=DirectoryLoader(data,
                           glob='*.pdf',loader_cls=PyPDFLoader)
    documents=loader.load()
    return documents

documents=load_pdf(data=DATA_PATH)
#print(len(documents))


#STEP2 CREATE CHUNKS
def create_chunks(ext_data):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=50)
    text_chunk=text_splitter.split_documents(ext_data)
    return text_chunk

text_chunk=create_chunks(ext_data=documents)
#print(len(text_chunk))

#STEP 3 CREATE VECTORS
def get_embed_model():
    embedding_model=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-V2")
    return embedding_model
embedding_model=get_embed_model()
   
 
 #STEP 4 Store vectos embed in FAISS
DB_FAISS_PATH="vectorstore/db_faiss"
db=FAISS.from_documents(text_chunk,embedding_model)
db.save_local(DB_FAISS_PATH)

