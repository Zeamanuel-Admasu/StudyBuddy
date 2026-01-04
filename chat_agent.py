import ast
import operator as op
from dotenv import load_dotenv

from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings  # pip install -U langchain-huggingface
from langchain_community.vectorstores import FAISS
from langchain_core.tools import tool
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import ChatPromptTemplate


# ---- Safe calculator tool (supports + - * / ** and parentheses) ----
_ALLOWED_OPS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.Pow: op.pow,
    ast.USub: op.neg,
}

def _eval(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp):
        return _ALLOWED_OPS[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp):
        return _ALLOWED_OPS[type(node.op)](_eval(node.operand))
    raise ValueError("Unsupported expression. Use only numbers and + - * / ** ( )")

@tool
def calculator(expression: str) -> str:
    """Evaluate a math expression like 2*(3+4)**2. Use ** for power."""
    expression = expression.replace("^", "**")
    tree = ast.parse(expression, mode="eval")
    return str(_eval(tree.body))


@tool
def search_notes(query: str) -> str:
    """Search the local FAISS index and return the most relevant context."""
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    docs = retriever.invoke(query)
    return "\n\n".join(d.page_content for d in docs)


def main():
    load_dotenv()

    tools = [search_notes, calculator]

    # IMPORTANT: set this to EXACTLY what `ollama list` shows
    model = ChatOllama(model="llama3.1:8b", temperature=0)
    history = []
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are StudyBuddy.\n"
         "Use tools when helpful:\n"
         "- Use search_notes for questions about the user's documents.\n"
         "- Use calculator for math.\n"
         "If the notes do not contain enough info, say you do not know.\n"
         "Be concise.\n"),
        ("human",
        "Conversation so far:\n{history}\n\n"
        "Question: {input}\n\n"
        "You have access to these tools:\n{tools}\n\n"
        "Tool names:\n{tool_names}\n\n"
        "Use this format:\n"
        "Thought: ...\n"
        "Action: tool_name\n"
        "Action Input: ...\n"
        "Observation: ...\n"
        "(repeat as needed)\n"
        "Final: ...")

    ])
    agent = create_react_agent(model, tools, prompt)
    executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    print("StudyBuddy Agent (type 'exit' to quit)\n")
       

    
    while True:
        q = input("You: ").strip()
        if q.lower() in {"exit", "quit"}:
            break
        history_text = "\n".join(history[-8:])  

        result = executor.invoke({"input": q, "history": history_text})
        print("\nStudyBuddy:", result["output"], "\n")


if __name__ == "__main__":
    main()
