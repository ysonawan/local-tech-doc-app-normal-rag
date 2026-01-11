import streamlit as st
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from vector import retriever

# Page configuration
st.set_page_config(
    page_title="Tech Doc RAG Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Helper function to format retriever output as string
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Initialize model and chains
@st.cache_resource
def load_rag_chain():
    """Load RAG chain once and cache it"""
    model = OllamaLLM(model="llama3.2")

    TECH_DOC_SYSTEM_PROMPT = """
You are a Tech Documentation AI Agent.

You must answer strictly from the provided documentation context.
If the context is empty or insufficient, say so clearly.
Never hallucinate APIs, configs, or behaviors.
"""

    TECH_DOC_RAG_PROMPT = """Based on the following documentation:

{context}

Answer this question: {question}"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", TECH_DOC_SYSTEM_PROMPT),
        ("human", TECH_DOC_RAG_PROMPT)
    ])

    rag_chain = (
        {
            "context": retriever,
            "question": RunnablePassthrough()
        }
        | prompt
        | model
        | StrOutputParser()
    )

    retrieval_chain = retriever

    return rag_chain, retrieval_chain

# Load chains
rag_chain, retrieval_chain = load_rag_chain()

# Title and header
st.title("🤖 Tech Documentation RAG Agent")
st.markdown("---")
st.markdown("Ask questions about **Spring Boot 4.0** and **MongoDB 8.0** documentation")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    st.markdown("### About")
    st.markdown("""
    This RAG (Retrieval-Augmented Generation) system retrieves relevant documentation
    and uses LLaMA 3.2 to generate accurate answers.
    
    **Sources:**
    - Spring Boot 4.0 Release Notes
    - Spring Boot 4.0 Migration Guide
    - MongoDB 8.0 Release Notes
    - MongoDB 8.0 Upgrade Guide
    """)

    st.markdown("### Features")
    st.markdown("""
    ✅ Real-time document retrieval
    ✅ Accurate context-based answers
    ✅ Source attribution
    ✅ Multi-document support
    """)

# Main content area
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("📝 Ask a Question")
    question = st.text_area(
        "Enter your question:",
        height=120,
        placeholder="e.g., Is there anything about JmsClient in Spring Boot 4.0?",
        label_visibility="collapsed"
    )

with col2:
    st.subheader("💡 Options")
    show_context = st.checkbox("Show Retrieved Context", value=True)
    show_full_response = st.checkbox("Show Full Response", value=True)

# Process question
if question.strip():
    if st.button("🔍 Search & Answer", use_container_width=True, type="primary"):
        try:
            with st.spinner("🔄 Retrieving documents and generating answer..."):
                # Get context
                context_docs = retrieval_chain.invoke(question)

                # Get answer
                answer = rag_chain.invoke(question)

            # Display results
            st.markdown("---")

            if show_context:
                st.subheader("📚 Retrieved Context")
                with st.expander("View Retrieved Documentation", expanded=True):
                    st.markdown("""
                    <div style="background-color: #f0f2f6; padding: 15px; border-radius: 8px; border-left: 4px solid #1f77e4;">
                    """, unsafe_allow_html=True)
                    st.markdown(context_docs)
                    st.markdown("</div>", unsafe_allow_html=True)

            if show_full_response:
                st.subheader("💡 Answer")
                with st.expander("View Full Answer", expanded=True):
                    st.markdown("""
                    <div style="background-color: #d4edda; padding: 15px; border-radius: 8px; border-left: 4px solid #28a745;">
                    """, unsafe_allow_html=True)
                    st.markdown(answer)
                    st.markdown("</div>", unsafe_allow_html=True)

            st.success("✅ Answer generated successfully!")

        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            st.info("Make sure Ollama is running with the llama3.2 model")

else:
    st.info("👆 Enter a question above to get started!")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #888;">
    <small>Powered by LangChain + Ollama + Chroma</small>
</div>
""", unsafe_allow_html=True)

