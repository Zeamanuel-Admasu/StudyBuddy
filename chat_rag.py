import os
import re
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings  # pip install -U langchain-huggingface
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama


STOPWORDS = {
    "the", "a", "an", "and", "or", "to", "of", "in", "on", "for", "with", "is", "are",
    "was", "were", "it", "this", "that", "i", "you", "we", "they", "he", "she", "as",
    "at", "by", "from", "be", "do", "does", "did", "what", "why", "how", "when", "where"
}

def retrieval_is_weak(question: str, context: str) -> bool:
    """
    Heuristic: if the question shares almost no meaningful words with retrieved context,
    treat retrieval as weak.
    """
    q_words = {w for w in re.findall(r"[a-zA-Z]{3,}", question.lower()) if w not in STOPWORDS}
    c_words = {w for w in re.findall(r"[a-zA-Z]{3,}", context.lower()) if w not in STOPWORDS}

    # Greetings / very short inputs -> weak
    if len(q_words) == 0:
        return True

    overlap_ratio = len(q_words & c_words) / max(1, len(q_words))
    return overlap_ratio < 0.15


def _format_source(meta: dict) -> str:
    src = meta.get("source", "unknown")
    src = os.path.basename(src)
    page = meta.get("page", None)  # PyPDFLoader usually uses 0-based page index
    if page is None:
        return src
    return f"{src} (page {page + 1})"


def build_context_and_sources(docs):
    source_lines = []
    context_parts = []
    for i, d in enumerate(docs, start=1):
        tag = f"[S{i}]"
        src = _format_source(d.metadata)
        source_lines.append(f"{tag} {src}")
        context_parts.append(f"{tag}\n{d.page_content}")
    context = "\n\n".join(context_parts)
    sources_text = "\n".join(source_lines)
    return context, source_lines, sources_text


def main():
    load_dotenv()

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    model = ChatOllama(model="llama3.1:8b", temperature=0)

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are StudyBuddy. Answer using ONLY the provided context. "
         "If the context is not enough, say you do not know.\n\n"
         "CITATION RULES:\n"
         "1) You MUST cite sources using ONLY the tags listed under 'Available sources'.\n"
         "2) Do NOT invent new tags.\n"
         "3) Put citations at the end of the sentence they support, like: ... [S1]"),
        ("human",
         "Conversation so far:\n{history}\n\n"
         "Available sources:\n{sources}\n\n"
         "Question: {question}\n\n"
         "Context:\n{context}")
    ])

    history = []
    chain = prompt | model | StrOutputParser()

    print("StudyBuddy (type 'exit' to quit)\n")

    while True:
        q = input("You: ").strip()
        if q.lower() in {"exit", "quit"}:
            break

        # 1) Retrieve
        docs = retriever.invoke(q)
        context, source_lines, sources_text = build_context_and_sources(docs)

        # 2) Weak retrieval -> ask clarification, then retrieve again
        if retrieval_is_weak(q, context):
            clar = input("StudyBuddy: I’m not sure what to look for in your notes. "
                         "Can you rephrase or give a keyword/name from the document?\nYou: ").strip()
            if clar.lower() in {"exit", "quit"}:
                break

            improved_query = f"{q}\n\nClarification: {clar}"
            docs = retriever.invoke(improved_query)
            context, source_lines, sources_text = build_context_and_sources(docs)

        # 3) Answer
        history_text = "\n".join(history[-8:])

        ans = chain.invoke({
            "question": q,
            "context": context,
            "history": history_text,
            "sources": sources_text,
        })

        print("\nStudyBuddy:", ans, "\n")
        print("Sources:")
        for line in source_lines:
            print(" -", line)
        print()

        history.append(f"You: {q}")
        history.append(f"StudyBuddy: {ans}")


if __name__ == "__main__":
    main()
