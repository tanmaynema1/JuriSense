import streamlit as st
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_community import embeddings
from dotenv import load_dotenv
from langchain_community.llms import Ollama
import google.generativeai as genai
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import CharacterTextSplitter 
from PyPDF2 import PdfReader
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
import time
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    embedding=embeddings.ollama.OllamaEmbeddings(model='mistral')
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embedding)
    return vectorstore

def get_conversational_chain(vectorstore):
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "Answer is not available in the Context", don't provide the wrong answer.\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro",
                                   client=genai,
                                   temperature=0.3,
                                   )
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    prompt = PromptTemplate(template=prompt_template,
                            input_variables=["context", "question"])
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=model,
        retriever= vectorstore.as_retriever(),
        memory=memory,
        prompt=prompt
    )
    return conversation_chain

def clear_chat_history():
    st.session_state.messages = [
        {"role": "assistant", "content": "upload some pdfs and ask me a question"}]

def handle_userinput(user_question):
    if st.session_state.conversation:
        try:
            with st.spinner("Thinking..."):
                response = st.session_state.conversation({'question': user_question})
                st.session_state.chat_history = response['chat_history']

                for i, message in enumerate(st.session_state.chat_history):
                    if i % 2 == 0:
                        st.write(user_template.replace(
                            "{{MSG}}", message.content), unsafe_allow_html=True)
                    else:
                        st.write(bot_template.replace(
                            "{{MSG}}", message.content), unsafe_allow_html=True)
        except Exception as e:
            st.write(f"An error occurred: {str(e)}")
            if "Recitation Error" in str(e):
                st.write("Switching to Ollama Mistral model...")
                model_local = Ollama(model="mistral")
                response = model_local(user_question)
                st.write(bot_template.replace("{{MSG}}", response), unsafe_allow_html=True)
            else:
                st.write("Gemini model timed out. Switching to Ollama Mistral model...")
                model_local = Ollama(model="mistral")
                response = model_local(user_question)
                st.write(bot_template.replace("{{MSG}}", response), unsafe_allow_html=True)
    else:
        st.write(bot_template.replace(
            "{{MSG}}", "These details are not mentioned in my knowledge base"), unsafe_allow_html=True)

            
def main():
    load_dotenv()
    st.set_page_config(page_title="**JuriSense**",
                       page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("**JuriSense**: Your Education Companion (Law)")
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                # get pdf text
                raw_text = get_pdf_text(pdf_docs)

                # get the text chunks
                text_chunks = get_text_chunks(raw_text)

                # create vector store
                vectorstore = get_vectorstore(text_chunks)

                # create conversation chain
                st.session_state.conversation = get_conversation_chain(
                    vectorstore)


if __name__ == '__main__':
    main()


