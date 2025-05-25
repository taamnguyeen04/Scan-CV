import cv2
import pytesseract
from PIL import Image
from collections import defaultdict
from langchain_community.document_loaders import PDFPlumberLoader
from langchain.schema import Document
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains import RetrievalQA
import streamlit as st
import tempfile
import os
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def process_image_to_text(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    ocr_data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
    result_dict = defaultdict(list)

    for i in range(len(ocr_data["text"])):
        if ocr_data["text"][i].strip() and int(ocr_data["conf"][i]) > 60:
            key = (
                ocr_data["page_num"][i],
                ocr_data["block_num"][i],
                ocr_data["par_num"][i],
                ocr_data["line_num"][i],
            )
            result_dict[key].append(ocr_data["text"][i])

    lines = [" ".join(words) for words in result_dict.values()]
    full_text = "\n".join(lines)
    return full_text


@st.cache_resource
def build_qa_chain(uploaded_file):
    filetype = uploaded_file.type
    temp_path = os.path.join(tempfile.gettempdir(), uploaded_file.name)

    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getvalue())

    # === Xử lý file PDF ===
    if filetype == "application/pdf":
        loader = PDFPlumberLoader(temp_path)
        docs = loader.load()
    # === Xử lý ảnh: JPG, PNG ===
    elif filetype in ["image/jpeg", "image/png"]:
        text = process_image_to_text(temp_path)
        if not text.strip():
            raise ValueError("Không trích xuất được văn bản từ ảnh.")
        docs = [Document(page_content=text)]
    else:
        raise ValueError("Định dạng tệp không được hỗ trợ.")
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