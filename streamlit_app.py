#Research & Development App - OS Natural Language Interface

#Modules
import streamlit as st
import os
import google.generativeai as ggi
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
from langchain.chains.question_answering import load_qa_chain
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.schema.runnable import RunnableMap
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_core.prompts import ChatPromptTemplate
from pathlib import Path
import os 

#Variables to hold our different documents to be used.
fileTroy = "prompt_answer.csv"
fileOS = "RAGDocuments/prompt_OS_answer.csv"

#Get API Key from Secrets file into a variable.
apikey = st.secrets["API_KEY"]

#Add api key as OS environment - Google Genai prefers it
os.environ["GOOGLE_API_KEY"] = apikey

#A function to load a document.
def DocLoader(fileName):    
   rootPath = '/workspaces/NLP-RD2'
   loader = CSVLoader(Path(rootPath,fileName), csv_args={'delimiter':','})
   return loader.load()

#A function to split a document.
#Chunking sizes chosen should cover entire lines and overlap parts of contiguous lines
def DocSplitter(document):
   splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=200)
   splitdocs = splitter.split_documents(document)
   #items = []
   #for doc in splitdocs:
    #items += doc      
   #strList = ','.join(items)
   #return strList
   return str(splitdocs)

#Load our documents used for RAG
loadedTroy = DocLoader(fileTroy)
#loadedOS = DocLoader(fileOS)

#Split our documents
troy_Split = DocSplitter(loadedTroy)
#OS_Split = DocSplitter(loadedOS)

#Create an embeddings object.
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001",google_api_key=apikey)

#Create a vector store from the text in the CSV file
vectorstore = DocArrayInMemorySearch.from_texts(troy_Split,embedding=embeddings)

retriever = vectorstore.as_retriever()

#Text to display status to the user
headerDisplay = "Hello"
detailDisplay = "Please ask a question above"

st.title("Gemini assistant & :red[NLP OS I/F R&D]")

#Create Gemini AI object. Apply the Gemini API Key. Set a loose temperature.
model = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=apikey,temperature=0.9,
                                  convert_system_message_to_human=True)

#Prompt the user to input their request.
userQuestion = st.text_area("You can ask general questions, questions about Troy University, and in the future interface with your OS! Press **CTRL+Enter** to send your question.")

#Clear the section which displays results from Gemini.
responseTitle = st.empty()
responseTitle.write("")
responseBody = st.empty()
responseBody.write("")

#The prompt template and prompt.
template = """Answer the question based on the following context:
{context}

#Question: {question}
#"""

prompt = ChatPromptTemplate.from_template(template)

#Functionality to perform the communication with the API and then display the results.
if userQuestion:
    responseTitle.write("Processing")
    responseBody.write("")   
    chain = RunnableMap({
      "context": lambda x: retriever.get_relevant_documents(x["question"]),
      "question": lambda x: x["question"]
    }) | prompt | model      
    output = chain.invoke({"question": userQuestion})
    responseTitle.write("")
    responseBody.write(output.content)




