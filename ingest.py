import os
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

def load_docs(data_dir: str):
    docs = []
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Missing folder: {data_dir}")
    for name in os.listdir(data_dir):
        path = os.path.join(data_dir, name)
        if name.lower().endswith(".txt"):
            docs.extend(TextLoader(path, encoding="utf-8").load())
        elif name.lower().endswith(".pdf"):
            docs.extend(PyPDFLoader(path).load())
    if not docs:
        raise RuntimeError(
            "No documents found. Put at least one .txt or .pdf file inside the 'data/' folder."
        )
    return docs
def main():
    load_dotenv()
    data_dir = "data"

    docs = load_docs(data_dir)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local("faiss_index")
    print(f"Saved FAISS index with {len(chunks)} chunks.")
if __name__ == "__main__":
    main()
