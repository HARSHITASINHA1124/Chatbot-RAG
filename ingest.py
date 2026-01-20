import wikipedia
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

os.makedirs("data", exist_ok=True)
os.makedirs("embeddings", exist_ok=True)

topics = [
    "Artificial Intelligence",
    "Machine Learning",
    "Deep Learning",
    "Neural Networks"
]

docs = []

for topic in topics:
    try:
        print(f"Fetching Wikipedia page for: {topic}")
        search_results = wikipedia.search(topic)
        if not search_results:
            continue
        page = wikipedia.page(search_results[0], auto_suggest=False)
        docs.append(page.content)
    except Exception as e:
        print(f"Skipping {topic}: {e}")

with open("data/wiki_docs.txt", "w", encoding="utf-8") as f:
    f.write("\n\n".join(docs))

with open("data/wiki_docs.txt", "r", encoding="utf-8") as f:
    text = f.read()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = splitter.split_text(text)
print(f"Total chunks created: {len(chunks)}")

# FREE embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = FAISS.from_texts(chunks, embeddings)
vectorstore.save_local("embeddings/faiss_index")

print("✅ FREE ingestion complete")
