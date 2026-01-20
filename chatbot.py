from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from transformers import pipeline

# Embeddings (same as ingest)
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = FAISS.load_local(
    "embeddings/faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# Free LLM
hf_pipeline = pipeline(
    "text2text-generation",
    model="google/flan-t5-base",
    max_length=256
)

llm = HuggingFacePipeline(pipeline=hf_pipeline)

# Prompt
prompt = PromptTemplate.from_template(
    """Use the context below to answer the question.
If the answer is not in the context, say "I don't know".

Context:
{context}

Question:
{question}

Answer:"""
)

# RAG Chain (NEW STYLE)
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
)

def ask_question(question: str):
    return rag_chain.invoke(question)
