import streamlit as st
import google.generativeai as genai
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
import os

# --- Configuration ---
# Load API key from Streamlit secrets
try:
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
except KeyError:
    st.error("Google API Key not found in Streamlit secrets. Please add it to .streamlit/secrets.toml")
    st.stop()

# Persistent directory for ChromaDB
PERSIST_DIRECTORY = 'data/chroma_db'
if not os.path.exists(PERSIST_DIRECTORY):
    os.makedirs(PERSIST_DIRECTORY)

# --- Functions ---

@st.cache_resource
def get_vector_store(documents):
    """Generates and persists a vector store from documents."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = Chroma.from_documents(
        documents=texts,
        embedding=embeddings,
        persist_directory=PERSIST_DIRECTORY
    )
    vector_store.persist()
    st.success("Vector store created and persisted!")
    return vector_store

@st.cache_resource
def load_vector_store():
    """Loads an existing vector store from the persistent directory."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    if os.path.exists(PERSIST_DIRECTORY) and os.listdir(PERSIST_DIRECTORY):
        vector_store = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings)
        st.success("Existing vector store loaded!")
        return vector_store
    return None

def get_qa_chain(vector_store):
    """Creates and returns a RetrievalQA chain."""
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    qa_chain = RetrievalQA.from_chain_type(
        llm=model,
        chain_type="stuff",
        retriever=vector_store.as_retriever(),
        return_source_documents=True
    )
    return qa_chain

def process_uploaded_files(uploaded_files):
    """Processes uploaded files and extracts text."""
    all_documents = []
    for uploaded_file in uploaded_files:
        file_extension = uploaded_file.name.split('.')[-1].lower()
        file_path = os.path.join("data", uploaded_file.name)

        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        if file_extension == "pdf":
            loader = PyPDFLoader(file_path)
            all_documents.extend(loader.load())
        elif file_extension == "txt":
            loader = TextLoader(file_path)
            all_documents.extend(loader.load())
        else:
            st.warning(f"Skipping unsupported file type: {uploaded_file.name}")
    return all_documents

# --- Streamlit UI ---

st.set_page_config(page_title="RAG Research Query Bot", layout="wide")
st.title("📚 RAG Research Query Bot with Gemini")

# Sidebar for options
st.sidebar.header("Options")
selected_option = st.sidebar.radio(
    "Choose an action:",
    ("Upload Research Papers", "Ask Query (if papers uploaded)")
)

# Initialize session state for vector store
if "vector_store" not in st.session_state:
    st.session_state.vector_store = load_vector_store()

if selected_option == "Upload Research Papers":
    st.header("Upload Research Papers (PDF or TXT)")
    uploaded_files = st.file_uploader(
        "Choose PDF or TXT files",
        type=["pdf", "txt"],
        accept_multiple_files=True
    )

    if uploaded_files:
        if st.button("Process Papers"):
            with st.spinner("Processing papers and creating vector store..."):
                os.makedirs("data", exist_ok=True) # Ensure 'data' directory exists
                documents = process_uploaded_files(uploaded_files)
                if documents:
                    st.session_state.vector_store = get_vector_store(documents)
                    st.success(f"Successfully processed {len(uploaded_files)} paper(s)!")
                else:
                    st.warning("No supported documents found to process.")
    else:
        st.info("Upload your research papers to enable querying.")

elif selected_option == "Ask Query (if papers uploaded)":
    st.header("Ask a Question About Uploaded Papers")

    if st.session_state.vector_store:
        user_query = st.text_area("Enter your question here:", height=100)
        if st.button("Get Answer"):
            if user_query:
                with st.spinner("Finding the answer..."):
                    qa_chain = get_qa_chain(st.session_state.vector_store)
                    response = qa_chain({"query": user_query})
                    st.subheader("Answer:")
                    st.write(response["result"])
                    st.subheader("Source Documents:")
                    for i, doc in enumerate(response["source_documents"]):
                        st.write(f"*Document {i+1}:*")
                        st.write(f"  Source: {doc.metadata.get('source', 'N/A')}")
                        st.write(f"  Page: {doc.metadata.get('page', 'N/A')}")
                        with st.expander("View Content"):
                            st.text(doc.page_content)
            else:
                st.warning("Please enter a question.")
    else:
        st.warning("Please upload and process research papers first using the 'Upload Research Papers' option.")

st.sidebar.markdown("---")
st.sidebar.info("This bot uses Google's Gemini Pro for answering queries and Gemini Embeddings for vectorization.")
