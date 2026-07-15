import os
import streamlit as st

from utils.pdf_loader import load_pdf, split_documents
from utils.vector_store import create_vector_store, get_retriever
from utils.llm import llm
from utils.rag import build_chain

st.set_page_config(
    page_title="Research Paper Assistant",
    layout="wide"
)

st.title("📄 Research Paper Question Answering using RAG")

st.write(
    "Upload a research paper and ask questions about it."
)

# ---------------- Upload PDF ---------------- #

uploaded_file = st.file_uploader(
    "Upload Research Paper",
    type=["pdf"]
)

# Create uploads folder
os.makedirs("uploads", exist_ok=True)

# ---------------- Process PDF ---------------- #

if uploaded_file is not None:

    pdf_path = os.path.join(
        "uploads",
        uploaded_file.name
    )

    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success("✅ PDF Uploaded Successfully")

   
    if (
        "uploaded_filename" not in st.session_state
        or st.session_state.uploaded_filename != uploaded_file.name
    ):

        st.session_state.uploaded_filename = uploaded_file.name

        with st.spinner("📖 Reading PDF..."):
            documents = load_pdf(pdf_path)

        with st.spinner("✂️ Splitting into chunks..."):
            chunks = split_documents(documents)

        st.success(f"✅ Created {len(chunks)} chunks.")

        with st.spinner("🧠 Creating embeddings and Chroma database..."):
            vector_db = create_vector_store(chunks)

        retriever = get_retriever(vector_db)

        st.session_state.qa_chain = build_chain(
            llm,
            retriever
        )

        st.success("✅ Research paper indexed successfully!")

# ---------------- Question Answering ---------------- #

if "qa_chain" in st.session_state:

    st.divider()

    st.subheader("💬 Ask Questions")

    question = st.text_input(
        "Enter your question about the uploaded paper"
    )

    if st.button("Generate Answer"):

        if question.strip() == "":
            st.warning("Please enter a question.")
        else:

            with st.spinner("🔍 Retrieving relevant information..."):

                response = st.session_state.qa_chain.invoke(question)

            st.subheader("📌 Answer")

            st.write(response)

else:

    st.info("📄 Upload a research paper to begin.")