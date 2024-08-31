import os
import streamlit as st
import pickle
import time
from langchain import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

from dotenv import load_dotenv
load_dotenv()

st.title("RockyBot: News Research Tool 📈")
st.sidebar.title("News Article URLs")

links = [st.sidebar.text_input(f"URL {i+1}") for i in range(3)]

process_button = st.sidebar.button("Process URLs")
faiss_index_path = "faiss_index.pkl"

display_area = st.empty()
language_model = OpenAI(temperature=0.9, max_tokens=500)

if process_button:
    loader = UnstructuredURLLoader(urls=links)
    display_area.text("Loading data... Please wait...✅✅✅")
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=1000
    )
    display_area.text("Splitting text... Please wait...✅✅✅")
    chunks = text_splitter.split_documents(documents)
    
    embeddings_model = OpenAIEmbeddings()
    faiss_index = FAISS.from_documents(chunks, embeddings_model)
    display_area.text("Building embeddings... Please wait...✅✅✅")
    time.sleep(2)

    with open(faiss_index_path, "wb") as file:
        pickle.dump(faiss_index, file)

search_query = display_area.text_input("Enter your question: ")
if search_query:
    if os.path.exists(faiss_index_path):
        with open(faiss_index_path, "rb") as file:
            index = pickle.load(file)
            retrieval_chain = RetrievalQAWithSourcesChain.from_llm(llm=language_model, retriever=index.as_retriever())
            response = retrieval_chain({"question": search_query}, return_only_outputs=True)
            
            st.header("Answer")
            st.write(response["answer"])

            sources = response.get("sources", "")
            if sources:
                st.subheader("Sources:")
                source_list = sources.split("\n")
                for source in source_list:
                    st.write(source)
