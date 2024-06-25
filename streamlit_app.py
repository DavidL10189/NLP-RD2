#Research & Development App - OS Natural Language Interface

#Modules
import streamlit as st
import os
import google.generativeai as ggi
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
#from langchain_community.document_loaders.csv_loader import CSVLoader
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


#Variables to hold our different documents to be used.
fileTroy = "/mount/src/nlp-rd2/InputDocs/prompt_answer.csv"
fileOS = "/mount/src/nlp-rd2/InputDocs/prompt_OS_answer.csv"

#Get API Key from Secrets file into a variable.
apikey = st.secrets["API_KEY"]

allInputLines = []

#Function to read lines from CSV files
def ReadCSV(fileName):
    with open ((fileName),'r') as file:
        lines = file.read()        
        return lines.split("\n\n")
        

allInputLines = ReadCSV(fileTroy)
allInputLines += ReadCSV(fileOS)


#Create an embeddings object.
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001",google_api_key=apikey)

vectorstore = DocArrayInMemorySearch.from_texts(
allInputLines,
embedding=embeddings
)

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
template = """ 
Question: {question}
Context: {context}
"""

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