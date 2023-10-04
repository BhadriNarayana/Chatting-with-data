import streamlit as st
import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader

from langchain.text_splitter import CharacterTextSplitter

from langchain.embeddings import HuggingFaceHubEmbeddings

from langchain.vectorstores import FAISS 

from langchain.memory import ConversationBufferMemory

from langchain.chains import ConversationalRetrievalChain

from langchain.llms import HuggingFaceHub


from htmlTemp import css, bot_template, user_template


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
    
def get_conversation_chain(vs):
    llm = HuggingFaceHub(repo_id='google/flan-t5-base', huggingfacehub_api_token =os.getenv('HF_HUB_KEY'))
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversational_chain = ConversationalRetrievalChain.from_llm(
        llm = llm,
        retriever = vs.as_retriever(),
        memory = memory

    )

    return conversational_chain

def handle_userinput(user_question):
    if user_question and st.session_state.lf:
        resp = st.session_state.conversation({'question': user_question})
        # st.write(resp)
        st.session_state.chat_history = resp['chat_history']


        for i, message in enumerate(st.session_state.chat_history):
            if i%2==0:
                st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
            
            else:
                st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
    
    else:
        st.write("Please load the files in the sidebar and then click on the 'Process' button to process your data and then start chatting here")    
            

def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with pdf data", page_icon=":books:")

    st.write(css, unsafe_allow_html=True)

    st.session_state.lf = False

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history"  not in st.session_state:
        st.session_state.chat_history = None

    
    st.header("Chat with your pdfs! :books:")

    user_question = st.text_input("Ask a question abot your documents:")

    if user_question:
        handle_userinput(user_question)
    

    st.write(user_template.replace("{{MSG}}", "Hello from user"), unsafe_allow_html=True)
    st.write(bot_template.replace("{{MSG}}", "Hello from bot"), unsafe_allow_html=True)

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
                #st.write(vs)

                st.session_state.conversation = get_conversation_chain(vs)
                st.session_state.lf = True
                  




    


if __name__ == '__main__':
    main()