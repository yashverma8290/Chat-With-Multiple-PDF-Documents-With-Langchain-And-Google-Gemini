import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS  # ✅ Updated import
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from google.oauth2 import service_account
import google.generativeai as genai

# ✅ Path to your Google Gemini Service Account key JSON
GOOGLE_CREDENTIALS_PATH = r"C:\Users\verma\AIML\DEEP-LEARNING\Chat With Multiple PDF Documents With Langchain And Google Gemini Pro\gemini_key.json"

# ✅ Load credentials
credentials = service_account.Credentials.from_service_account_file(
    GOOGLE_CREDENTIALS_PATH,
    scopes=["https://www.googleapis.com/auth/cloud-platform"]
)

# ✅ Configure Gemini client with credentials
genai.configure(credentials=credentials)


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text.strip()
        except Exception as e:
            st.error(f"❌ Error reading {pdf.name}: {str(e)}")
    return text


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        credentials=credentials
    )
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details. 
    If the answer is not in the provided context, just say "answer is not available in the context".
    
    Context:\n{context}\n
    Question:\n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(
        model="gemini-pro",
        temperature=0.3,
        credentials=credentials
    )

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain


def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        credentials=credentials
    )

    # ✅ Check if index exists
    if not os.path.exists("faiss_index"):
        st.error("❌ FAISS index not found. Please upload and process PDFs first.")
        return

    try:
        new_db = FAISS.load_local("faiss_index", embeddings)
        docs = new_db.similarity_search(user_question)

        chain = get_conversational_chain()
        response = chain(
            {"input_documents": docs, "question": user_question},
            return_only_outputs=True
        )

        st.write("🧠 Reply:", response["output_text"])
    except Exception as e:
        st.error(f"❌ Error during search or response: {str(e)}")


def main():
    st.set_page_config("Chat PDF")
    st.header("💬 Chat with Multiple PDFs using Gemini")

    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("📂 Upload PDFs")
        pdf_docs = st.file_uploader(
            "Upload PDF Files and Click on 'Submit & Process'",
            accept_multiple_files=True
        )
        if st.button("Submit & Process"):
            if pdf_docs:
                with st.spinner("Processing..."):
                    raw_text = get_pdf_text(pdf_docs)
                    if raw_text:
                        text_chunks = get_text_chunks(raw_text)
                        get_vector_store(text_chunks)
                        st.success("✅ Processing complete!")
                    else:
                        st.warning("⚠️ No text extracted from PDFs.")
            else:
                st.warning("⚠️ Please upload at least one PDF.")


if __name__ == "__main__":
    main()
