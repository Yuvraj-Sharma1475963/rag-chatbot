from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from openai import OpenAI

load_dotenv()

# ── Step 1: Load PDF ──────────────────────────────────────────
loader = PyPDFLoader("sample.pdf")
pages = loader.load()
print(f"✅ PDF loaded — {len(pages)} pages found")

# ── Step 2: Split into chunks ─────────────────────────────────
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = splitter.split_documents(pages)
print(f"✅ Split into {len(chunks)} chunks")

# ── Step 3: Embeddings + FAISS ────────────────────────────────
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(chunks, embeddings)
print("✅ Vector database ready!")

# ── Step 4: Direct OpenAI (no LangChain needed) ───────────────
client = OpenAI()

def ask(question):
    # get relevant chunks from FAISS
    docs = vectorstore.similarity_search(question, k=3)
    context = "\n\n".join([d.page_content for d in docs])

    # call OpenAI directly
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "Answer questions using only the context provided. If the answer is not in the context, say I don't know."
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {question}"
            }
        ]
    )
    return response.choices[0].message.content

# ── Step 5: Test it ───────────────────────────────────────────
print("\n✅ RAG Chatbot ready!\n")

questions = [
    "What is the main topic of this document?",
    "What is the key advice given in this document?",
]

for q in questions:
    print(f"❓ {q}")
    print(f"🤖 {ask(q)}")
    print("-" * 60)