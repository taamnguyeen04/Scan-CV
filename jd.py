from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

# Load a PDF document and split it into chunks
file_path = "jd.pdf"  # Path of the document to be loaded
loader = PyPDFLoader(file_path)     # Initialize the pdf loader
documents = loader.load()           # Load the pdf document

# Initialize the recursive character text splitter
text_splitter = RecursiveCharacterTextSplitter(
    separators="",
    chunk_size=100,
    chunk_overlap=20
)

# Split the documents into chunks
chunks = text_splitter.split_documents(documents)

# Initialize the Hugging Face embedding model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# Store embeddings into the vector store
vector_store = FAISS.from_documents(
    documents=chunks,
    embedding=embeddings
)

# Retrieve relevant information using similarity score threshold
retriever = vector_store.as_retriever(
    search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.5}
)
docs = retriever.invoke("Ask any question from the document here")