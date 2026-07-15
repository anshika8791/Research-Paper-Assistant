from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_pdf(pdf_path):

    loader = PyMuPDFLoader(pdf_path)

    documents = loader.load()

    return documents

def split_documents(documents):

    splitter = RecursiveCharacterTextSplitter(

        chunk_size=1000,

        chunk_overlap=200

    )

    chunks = splitter.split_documents(documents)

    return chunks