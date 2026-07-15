from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Load embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


def create_vector_store(chunks):

    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="chroma_db"
    )

    return vector_db


def get_retriever(vector_db):

    retriever = vector_db.as_retriever(
        search_kwargs={"k": 4}
    )

    return retriever