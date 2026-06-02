

import re
import warnings
import logging
import os
import sys




import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
# from langchain_pymupdf4llm import PyMuPDF4LLMLoader
# from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import Chroma   # for old version of langchain, if needed
# from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# from pathlib import Path
import tempfile
# import magic
# import pdfplumber
# from io import BytesIO

import chromadb
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"




# ----------------------------------------------------------------------------------------
# 1. Block the "advisory" warnings from Transformers
from transformers import logging as transformers_logging

os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

# 2. Block the specific Python path-access warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="transformers")
warnings.filterwarnings('ignore', category=FutureWarning, module='transformers')
warnings.filterwarnings("ignore", message="Accessing `__path__` from")

# 2. Force the transformers logging library to only show critical errors
logging.getLogger("transformers").setLevel(logging.ERROR)
transformers_logging.set_verbosity_error()

# ----------------------------------------------------------------------------------------

st.set_page_config(page_title="Smart FAQ Chatbot", page_icon="", layout="centered")
st.title("Smart FAQ Chatbot")



load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    st.error("GROQ_API_KEY not found.")
    st.stop()


# Initialize session state variables

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "retriever" not in st.session_state:
    st.session_state.retriever = None
    
if "current_file" not in st.session_state:
    st.session_state.current_file = None
    
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
    
if "llm" not in st.session_state:
    st.session_state.llm = ChatGroq(
        api_key=groq_api_key,
        model="openai/gpt-oss-120b",
        temperature=0.7
    )    

 




class DocLoader:
    def __init__(self):
       
        self.detected_type = ""
        self.model = ""
        
        
               
       
        # File uploader form
        with st.form("upload"):
            self.uploaded_file = st.file_uploader(label = " Upload your document Allowed only: PDF, DOCX, TXT", )
            submitted = st.form_submit_button("Submit")
            
            if submitted:
                valid = self.check_file_type()
                if valid and self.uploaded_file is not None:
                    if st.session_state.current_file != self.uploaded_file.name:
                        st.session_state.current_file = self.uploaded_file.name
                        st.session_state.retriever = None
                        st.session_state.vector_store = None
                        
                        # Clean up old chat history
                        st.session_state.chat_history = []
                        
                        with st.spinner("Processing file..."):                     
                
                
                            self.save_file()
                            # st.success("File processing completed successfully!")
                            self.make_chunks()
                            self.embedding()
                            # st.success("Chunks embedded and stored in vector database")

                            self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 3})
                            
                            st.session_state.vector_store = self.vector_store
                            st.session_state.retriever = self.retriever
                            
                        st.success("File processing completed successfully!")
                    else:
                        st.info("Same file uploaded again. Skipping processing.")
    
        # Display chat history                   
        for role, message in st.session_state.chat_history:
            with st.chat_message(role):
                st.write(message)
                    
        # Chat input                
        self.query = st.chat_input("Ask a question about your document:")
        
        if self.query and st.session_state.retriever:
            self.retrieval_qa()            
                
       
                
        
    def check_file_type(self):
    
        ALLOWED_TYPES = {
            "application/pdf" : "PDF", 
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "DOCX",
            "text/plain": "TXT"
        }
        if self.uploaded_file is None:
                st.error("No file uploaded")
                return False

        self.detected_type = self.uploaded_file.type

        if self.detected_type not in ALLOWED_TYPES:
            st.error(
                f"Unsupported file type: '{self.detected_type}'. "
                f"Allowed only: PDF, DOCX, TXT"
            )
            return False

        st.success(f"Valid {ALLOWED_TYPES[self.detected_type]} file")

        return True
            
            
        
                    
   
    def save_file(self):                                  
            
        # preserve original extension
        suffix = f"_{self.uploaded_file.name}"
            
        # create temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            tmp_file.write(self.uploaded_file.getvalue())
        
            self.temp_path = tmp_file.name

        print("Saved path", self.temp_path)
        print(type(self.temp_path))
        # st.success(f"File uploaded and saved at {self.temp_path}")
        
        
       
                    
   
    
    def make_chunks(self):
        print("Making chunks...")
        
        try:        
            loader = PyMuPDFLoader(self.temp_path)
            docs = loader.load()
            st.success(f"Loaded {len(docs)} documents")               
        
        except Exception as e:
            st.error(f"File not loaded: {e}")
            return
        

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        ) 
    
        self.chunks = splitter.split_documents(docs)
        print(f"Number of chunks: {len(self.chunks)}")


    def embedding(self):
        
        print("Embedding chunks...")
        embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"})
        
        collection_name = self.uploaded_file.name.replace(".", "_").replace(" ", "_")
        
        # client = chromadb.PersistentClient(path="./data/chroma_db")
        # collection = client.create_collection(name=collection_name)
        
        
        # existing = [col.name for col in client.list_collections()]
        # if collection_name in existing:
        #     client.delete_collection(collection_name)
        #     print(f"Deleted existing collection: {collection_name}")
         
                
        self.vector_store = Chroma.from_documents(
            documents=self.chunks,
            # client=client,
            embedding=embedding_model,
            collection_name=collection_name,
            persist_directory="./data/chroma_db"
        )
       
        
            
    # def chat_model(self):
    #     print("Initializing chat model...")
    #     self.model = ChatGroq(
    #         api_key=groq_api_key,
    #         model="openai/gpt-oss-120b",
    #         temperature=0.7
    #     )
    #     print("Chat model initialized")

    
    
    
    def retrieval_qa(self):
        print("Setting up RetrievalQA chain...")
               
         
        
    
                
        # query = "What is deep learning?"
        
        
        system_prompt = (
            """You are a document QA assistant.
            Answer ONLY using the provided context.
            If the answer is not explicitly in the context, say:
            "I could not find that information in the document."
            Use metadata like source file and page number when relevant.
            Context: {context} """
        )
        
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", "{question}"),
            ]
        )
        
        def format_docs(docs):
            return "\n\n".join(
                f"Source: {doc.metadata}\n\n{doc.page_content}"
                for doc in docs
            )
        
                
        docs = st.session_state.retriever.invoke(self.query)
            
        context =  format_docs(docs)
        
        
        #  RAG_chain =  prompt | st.session_state.llm | StrOutputParser()
        
        RAG_chain =     (
            {
                "context": st.session_state.retriever | format_docs,
                "question": RunnablePassthrough()
            }
            | prompt
            | st.session_state.llm
            | StrOutputParser()
       )
        
        with st.chat_message("user"):
            st.write(self.query)
            
            
        with st.chat_message("assistant"):
            with st.spinner("Generating answer..."):
                try:
                    # result = RAG_chain.invoke({"question": self.query, "context": context})  tepada context ni prompt ichida berilganligi uchun, shunchaki question ni berish kifoya
                    result = RAG_chain.invoke(self.query)
                    st.session_state.chat_history.append(("user", self.query))
                    st.session_state.chat_history.append(("assistant", result))
                except Exception as e:
                    st.error(f"Error during RAG chain execution: {e}")
                    result = "Sorry, I couldn't generate an answer due to an error."
            st.write(result)        
        
        
        print("RetrievalQA chain executed successfully")
        
        
                
      
     
        
   


app = DocLoader()
