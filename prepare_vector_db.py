from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings

import os

# Khai bao bien
pdf_data_path = "pdf_data"
vector_db_path = "vectorstores/db_faiss"

# Ham 1. Tao ra vector DB tu 1 doan text
def create_db_from_text(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=512,
        chunk_overlap=50,
        length_function=len

    )
    chunks = text_splitter.split_text(text)

    if not chunks:
        raise ValueError("Không có đoạn văn bản nào để tạo vector!")

    # embedding_model = GPT4AllEmbeddings(model_file = "models/all-MiniLM-L6-v2-f16.gguf")
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    os.makedirs(os.path.dirname(vector_db_path), exist_ok=True)

    db = FAISS.from_texts(texts=chunks, embedding=embedding_model)
    db.save_local(vector_db_path)
    return db

with open("jd.txt", "r", encoding="utf-8") as file:
    text = file.read()

print(create_db_from_text(text))
