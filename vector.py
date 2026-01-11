
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

import os
import requests
from bs4 import BeautifulSoup
import shutil

# ----------------------------
# Configuration
# ----------------------------
TECH_DOC_URLS = [
    "https://github.com/spring-projects/spring-boot/wiki/Spring-Boot-4.0-Release-Notes",
    "https://github.com/spring-projects/spring-boot/wiki/Spring-Boot-4.0-Migration-Guide",
    "https://www.mongodb.com/docs/manual/release-notes/8.0/",
    "https://www.mongodb.com/docs/manual/release-notes/8.0-upgrade-replica-set/"
]
COLLECTION_NAME = "tech_docs"
DB_LOCATION = "./chroma_tech_docs"

embeddings = OllamaEmbeddings(model="mxbai-embed-large")

# Always clear the database on startup for fresh data
if os.path.exists(DB_LOCATION):
    shutil.rmtree(DB_LOCATION)
    print(f"🗑️  Cleared existing database: {DB_LOCATION}")

add_documents = True

# ----------------------------
# Scraper
# ----------------------------
def scrape_tech_doc(url: str) -> str:
    print(f" Scrapping url: {url}")
    response = requests.get(url, timeout=20)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    # Remove navigation, scripts, styles
    print(" Cleaning HTML content...")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()

    main_content = soup.find("main") or soup.body

    text = main_content.get_text(separator="\n")
    return text.strip()

# ----------------------------
# Build Documents
# ----------------------------
documents = []
ids = []

if add_documents:
    print(f"📄 Processing {len(TECH_DOC_URLS)} URL(s)...")

    for url_idx, url in enumerate(TECH_DOC_URLS):
        print(f"\n[{url_idx + 1}/{len(TECH_DOC_URLS)}] Processing: {url}")

        try:
            raw_text = scrape_tech_doc(url)

            # Optimal chunking for technical documentation
            # 1000 chars ≈ 150-200 words, captures complete technical concepts
            CHUNK_SIZE = 1000
            print(f" Splitting text into chunks (size={CHUNK_SIZE} chars)...")
            chunks = [
                raw_text[i:i + CHUNK_SIZE]
                for i in range(0, len(raw_text), CHUNK_SIZE)
            ]

            print(f" Embedding and adding {len(chunks)} chunks to documents...")
            for chunk_idx, chunk in enumerate(chunks):
                # Create unique ID combining URL index and chunk index
                unique_id = f"url{url_idx}_chunk{chunk_idx}"
                documents.append(
                    Document(
                        page_content=chunk,
                        metadata={
                            "source": url,
                            "url_index": url_idx,
                            "chunk": chunk_idx
                        }
                    )
                )
                ids.append(unique_id)

            print(f"✅ Added {len(chunks)} chunks from this URL")

        except Exception as e:
            print(f"❌ Error processing URL: {e}")
            continue

    print(f"\n📊 Total documents created: {len(documents)}")

# ----------------------------
# Vector Store
# ----------------------------
vector_store = Chroma(
    collection_name=COLLECTION_NAME,
    persist_directory=DB_LOCATION,
    embedding_function=embeddings
)

if add_documents:
    vector_store.add_documents(documents=documents, ids=ids)

# ----------------------------
# Retriever
# ----------------------------
retriever = vector_store.as_retriever(
    search_kwargs={"k": 5}
)
