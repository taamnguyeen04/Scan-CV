from langchain_community.document_loaders import PDFPlumberLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains import RetrievalQA
import streamlit as st


@st.cache_resource
def build_qa_chain(uploaded_file):
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.getvalue())
    loader = PDFPlumberLoader("temp.pdf")
    docs = loader.load()

    embedder = HuggingFaceEmbeddings()
    splitter = SemanticChunker(embedder)
    chunks = splitter.split_documents(docs)

    vector = FAISS.from_documents(chunks, embedder)
    retriever = vector.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    prompt_template = PromptTemplate.from_template("""
    Bạn đang đóng vai trò là một chuyên gia tuyển dụng có kinh nghiệm trong việc đánh giá mức độ phù hợp giữa hồ sơ ứng viên (CV) và bản mô tả công việc (JD). 
    Dựa trên các thông tin trong phần context dưới đây, hãy trả lời câu hỏi một cách trung thực và khách quan.

    - Nếu thông tin không đủ để trả lời, hãy trả lời "Tôi không chắc dựa trên thông tin hiện tại".
    - Không suy đoán hoặc bịa đặt thông tin không có trong context.

    Context: {context}
    Question: {question}
    Trả lời chi tiết và chính xác:
    """)

    llm = Ollama(model="deepseek-coder-v2:16b", temperature=0.1)
    llm_chain = LLMChain(llm=llm, prompt=prompt_template)
    combine_docs_chain = StuffDocumentsChain(
        llm_chain=llm_chain,
        document_variable_name="context",
        document_prompt=PromptTemplate(
            input_variables=["page_content", "source"],
            template="Context:\ncontent:{page_content}\nsource:{source}"
        )
    )
    qa_chain = RetrievalQA(combine_documents_chain=combine_docs_chain, retriever=retriever)
    return qa_chain

def extract_info(qa_chain, question):
    try:
        return qa_chain.run(question).strip()
    except:
        return ""