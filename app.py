import streamlit as st

def main():
    st.set_page_config(page_title="Chat with pdf data", page_icon=":books:")

    st.header("Chat with your pdfs! :books:")

    st.text_input("Ask a question abot your documents:")

    with st.sidebar:
        st.subheader("Your documents")
        st.file_uploader("Upload your PDFs here and click on 'Process'")
        st.button("Process")
    


if __name__ == '__main__':
    main()