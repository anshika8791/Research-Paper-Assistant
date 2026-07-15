from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


def format_docs(docs):
    """
    Convert retrieved Document objects into a single formatted string.
    """
    return "\n\n".join(doc.page_content for doc in docs)


def build_chain(llm, retriever):

    prompt = ChatPromptTemplate.from_template(
        """
You are an AI Research Assistant.

Use ONLY the information provided in the context to answer the user's question.

If the answer cannot be found in the context, respond with:

"I couldn't find this information in the uploaded research paper."

Context:
{context}

Question:
{question}

Answer:
"""
    )

    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain