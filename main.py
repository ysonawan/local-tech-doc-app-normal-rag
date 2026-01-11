from langchain_core.runnables import RunnablePassthrough
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from vector import retriever

# Helper function to format retriever output as string
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# 1. Model
model = OllamaLLM(model="llama3.2")

# 2. Prompt Template for Tech Documentation RAG
TECH_DOC_SYSTEM_PROMPT = """
You are a Tech Documentation AI Agent.

You must answer strictly from the provided documentation context.
If the context is empty or insufficient, say so clearly.
Never hallucinate APIs, configs, or behaviors.
"""

TECH_DOC_RAG_PROMPT = """Based on the following documentation:

{context}

Answer this question: {question}"""

# 3. Prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", TECH_DOC_SYSTEM_PROMPT),
    ("human", TECH_DOC_RAG_PROMPT)
])

# 4. Chain (RAG-ready) - returns the full context + answer
rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | prompt
    | model
    | StrOutputParser()
)

# 4b. Retrieval chain - to get context separately
retrieval_chain = retriever


# 5. Interactive loop
if __name__ == "__main__":
    while True:
        print("\n" + "-" * 80 + "\n")

        question = input("Enter your tech documentation question (or 'exit' to quit): ")
        if question.strip().lower() == "exit":
            break

        try:
            # Get the context first
            context = retrieval_chain.invoke(question)

            # Print context with formatting
            print("\n" + "=" * 80)
            print("📚 RETRIEVED CONTEXT:")
            print("=" * 80)
            print(context)
            print("=" * 80 + "\n")

            # Get and print the answer
            response = rag_chain.invoke(question)
            print("💡 ANSWER:")
            print("-" * 80)
            print(response)
            print("-" * 80)
        except Exception as e:
            print(f"❌ Error: {e}")
