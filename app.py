import streamlit as st
import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader

from langchain.text_splitter import CharacterTextSplitter

from langchain.embeddings import HuggingFaceHubEmbeddings

from langchain.vectorstores import FAISS 





def get_pdf_text(pdf_docs):
    text = ""

    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text+=page.extract_text()
    return text        

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size = 1000,
        chunk_overlap = 200,
        length_function = len
    )

    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(txt_chunks):

    embeddings = HuggingFaceHubEmbeddings(repo_id="sentence-transformers/all-MiniLM-L6-v2", huggingfacehub_api_token =os.getenv('HF_HUB_KEY'))

    vectorstore = FAISS.from_texts(txt_chunks, embedding=embeddings)

    return vectorstore
    

    
def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with pdf data", page_icon=":books:")

    st.header("Chat with your pdfs! :books:")

    st.text_input("Ask a question abot your documents:")

    with st.sidebar:
        st.subheader("Your documents")
        pdfs = st.file_uploader("Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        
        if st.button("Process"):
            with st.spinner("Processing"):
                raw_txt = get_pdf_text(pdfs)
                # st.write(raw_txt)
                text_chunks = get_text_chunks(raw_txt)
                #st.write(text_chunks)
                vs = get_vectorstore(text_chunks)
                st.write(vs)


    


if __name__ == '__main__':
    main()