import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from openai import OpenAI
import tempfile
import os

load_dotenv()
client = OpenAI()

st.set_page_config(page_title="RAG Chatbot", page_icon="🤖")
st.title("🤖 RAG Chatbot")
st.caption("Upload multiple PDFs and chat with all of them!")

# ── Session state ─────────────────────────────────────────────
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pdf_names" not in st.session_state:
    st.session_state.pdf_names = []

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.header("📄 Upload PDFs")
    uploaded_files = st.file_uploader(
        "Choose PDFs",
        type="pdf",
        accept_multiple_files=True
    )

    if uploaded_files and st.button("⚡ Process PDFs"):
        with st.spinner("Processing all PDFs..."):
            all_chunks = []
            names = []

            for uploaded_file in uploaded_files:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
                    f.write(uploaded_file.read())
                    tmp_path = f.name

                loader = PyPDFLoader(tmp_path)
                pages = loader.load()

                for page in pages:
                    page.metadata["source"] = uploaded_file.name

                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=500,
                    chunk_overlap=50
                )
                chunks = splitter.split_documents(pages)
                all_chunks.extend(chunks)
                names.append(f"📄 {uploaded_file.name} ({len(pages)} pages)")
                os.unlink(tmp_path)

            embeddings = OpenAIEmbeddings()
            st.session_state.vectorstore = FAISS.from_documents(
                all_chunks, embeddings
            )
            st.session_state.pdf_names = names
            st.session_state.messages = []

        st.success(f"✅ {len(uploaded_files)} PDF(s) ready!")

    if st.session_state.pdf_names:
        st.divider()
        st.markdown("**Loaded documents:**")
        for name in st.session_state.pdf_names:
            st.write(name)

    if st.session_state.vectorstore and st.button("🗑️ Clear all"):
        st.session_state.vectorstore = None
        st.session_state.messages = []
        st.session_state.pdf_names = []
        st.rerun()

# ── Chat ──────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if st.session_state.vectorstore is None:
    st.info("👈 Upload PDFs from the sidebar and click Process PDFs!")
else:
    if question := st.chat_input("Ask anything about your PDFs..."):
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # if follow-up question, combine with last answer for better search
                search_query = question
                if len(st.session_state.messages) > 2:
                    last_answer = st.session_state.messages[-2]["content"]
                    search_query = f"{question} {last_answer[:200]}"

                docs = st.session_state.vectorstore.similarity_search(search_query, k=4)
                context = "\n\n".join([d.page_content for d in docs])

                # ── Memory: build full history ────────────────
                history = [
                    {
                        "role": "system",
"content": """You are an intelligent document assistant. Follow these rules:

1. ALWAYS answer using only the provided document context.
2. If the answer is not in the context, say exactly: "I couldn't find this in the uploaded documents."
3. For follow-up questions, use conversation history to understand references like "the first one", "it", "that topic" etc.
4. Structure longer answers with clear points when listing multiple things.
5. Always be concise and direct — no unnecessary filler.
6. If asked to compare, find info from multiple documents and compare them clearly.
7. End answers with: "📄 Source details available below." when you used the context."""
                    },
                    {
                        "role": "user",
                        "content": f"Here is the relevant context:\n\n{context}"
                    },
                    {
                        "role": "assistant",
                        "content": "I have read the context. I will answer based on it."
                    }
                ]

                for msg in st.session_state.messages[:-1]:
                    history.append({
                        "role": msg["role"],
                        "content": msg["content"]
                    })

                history.append({
                    "role": "user",
                    "content": question
                })

                # streaming response — words appear one by one
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=history,
                    stream=True
                )
                
                answer = ""
                placeholder = st.empty()
                for chunk in response:
                    delta = chunk.choices[0].delta.content
                    if delta:
                        answer += delta
                        placeholder.markdown(answer + "▌")
                placeholder.markdown(answer)

                # ── Show source citations ─────────────────────────────────────
                with st.expander("📚 View Sources"):
                    seen = set()
                    for doc in docs:
                        source = doc.metadata.get("source", "Unknown")
                        page = doc.metadata.get("page", 0) + 1
                        key = f"{source}_p{page}"
                        if key not in seen:
                            seen.add(key)
                            st.markdown(f"**📄 {source}** — Page {page}")
                            st.caption(doc.page_content[:200] + "...")
                            st.divider()

                st.session_state.messages.append(
                    {"role": "assistant", "content": answer}
)