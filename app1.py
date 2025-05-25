import streamlit as st
import os
from PIL import Image
from langchain_community.document_loaders import PDFPlumberLoader, UnstructuredFileLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains import RetrievalQA
import matplotlib.pyplot as plt
import numpy as np
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
# ✅ Ngăn lỗi torch.classes do Streamlit Watcher
os.environ["STREAMLIT_WATCH_DISABLE"] = "true"

# ========== UI Customization ==========
st.set_page_config(page_title="CV & JD Matching", layout="centered")
logo = Image.open("logo2.jpg")
st.image(logo)
primary_color = "#1E90FF"
secondary_color = "#FF6347"
background_color = "#F5F5F5"
text_color = "#4561e9"

st.markdown(f"""
    <style>
    .stApp {{
        background-color: {background_color};
        color: {text_color};
    }}
    .stButton>button {{
        background-color: {primary_color};
        color: white;
        border-radius: 5px;
        border: none;
        padding: 10px 20px;
        font-size: 16px;
    }}
    .stFileUploader>div>div>div>button {{
        background-color: {secondary_color};
        color: white;
        border-radius: 5px;
        border: none;
        padding: 10px 20px;
        font-size: 16px;
    }}
    </style>
""", unsafe_allow_html=True)

st.title("\U0001F4C4 CV & JD")

# ========== Upload ==========
cv_file = st.file_uploader("Upload CV", type=["pdf", "docx", "png", "jpg", "jpeg"], key="cv")
jd_file = st.file_uploader("Upload JD", type=["pdf", "docx", "png", "jpg", "jpeg"], key="jd")


# ========== Embedding Model ==========
@st.cache_resource
def get_similarity_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("all-MiniLM-L6-v2")


def calculate_similarity(text1, text2):
    if not text1 or not text2:
        return 0.0
    from sentence_transformers import util
    model = get_similarity_model()
    emb1 = model.encode(text1, convert_to_tensor=True)
    emb2 = model.encode(text2, convert_to_tensor=True)
    return round(util.pytorch_cos_sim(emb1, emb2).item(), 2)


# ========== Radar Chart ==========
def draw_radar(scores_dict):
    labels = list(scores_dict.keys())
    values = list(scores_dict.values())
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    values += values[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.plot(angles, values, 'o-', linewidth=2)
    ax.fill(angles, values, alpha=0.25)
    ax.set_thetagrids(np.degrees(angles[:-1]), labels)
    ax.set_ylim(0, 100)
    ax.set_title("Biểu đồ đánh giá mức độ phù hợp", fontsize=16)
    st.pyplot(fig)


# ========== Prompt ==========
system_prompt = PromptTemplate.from_template("""
Bạn đang đóng vai trò là một chuyên gia tuyển dụng có kinh nghiệm trong việc đánh giá mức độ phù hợp giữa hồ sơ ứng viên (CV) và bản mô tả công việc (JD). 
Dựa trên các thông tin trong phần context dưới đây, hãy trả lời câu hỏi một cách trung thực và khách quan.

- Nếu thông tin không đủ để trả lời, hãy trả lời "Tôi không chắc dựa trên thông tin hiện tại".
- Không suy đoán hoặc bịa đặt thông tin không có trong context.

Context: {context}
Question: {question}
Trả lời chi tiết và chính xác:
""")


# ========== QA Builder ==========
@st.cache_resource
def build_qa_chain(uploaded_file):
    file_name = "temp_" + uploaded_file.name
    with open(file_name, "wb") as f:
        f.write(uploaded_file.getvalue())

    ext = uploaded_file.name.split(".")[-1].lower()
    if ext in ["jpg", "jpeg", "png"]:
        image = Image.open(file_name)
        extracted_text = pytesseract.image_to_string(image, lang='eng')
        # st.text_area("🖼 Văn bản trích xuất từ ảnh:", extracted_text, height=200)
        if not extracted_text.strip():
            st.warning("⚠️Không trích xuất được văn bản từ ảnh.")
            return lambda q: "Không có nội dung văn bản trong ảnh để phân tích."

        def qa_fn(question):
            llm = Ollama(model="deepseek-coder-v2:16b", temperature=0.1)
            prompt = system_prompt.format(context=extracted_text, question=question)
            return llm.invoke(prompt)

        return qa_fn

    # PDF or DOCX
    if ext == "pdf":
        loader = PDFPlumberLoader(file_name)
    else:
        loader = UnstructuredFileLoader(file_name)
    docs = loader.load()

    embedder = HuggingFaceEmbeddings()
    splitter = SemanticChunker(embedder)
    chunks = splitter.split_documents(docs)
    vector = FAISS.from_documents(chunks, embedder)
    retriever = vector.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    llm = Ollama(model="deepseek-coder-v2:16b", temperature=0.1)
    llm_chain = LLMChain(llm=llm, prompt=system_prompt)
    combine_docs_chain = StuffDocumentsChain(
        llm_chain=llm_chain,
        document_variable_name="context",
        document_prompt=PromptTemplate(
            input_variables=["page_content", "source"],
            template="Context:\ncontent:{page_content}\nsource:{source}"
        )
    )
    qa_chain = RetrievalQA(combine_documents_chain=combine_docs_chain, retriever=retriever)

    def qa_fn(question):
        return qa_chain.run(question).strip()

    return qa_fn


# ========== Evaluation ==========
if cv_file and jd_file:
    with st.spinner("🔍 Đang phân tích CV và JD..."):
        cv_qa = build_qa_chain(cv_file)
        jd_qa = build_qa_chain(jd_file)

        fields = {
            "Kỹ năng": "Liệt kê tất cả kỹ năng của ứng viên.",
            "Học vấn": "Trình độ học vấn của ứng viên là gì?",
            "Kinh nghiệm": "Ứng viên có kinh nghiệm gì?",
            "Bằng cấp": "Ứng viên có những bằng cấp nào?",
            "Vị trí mong muốn": "Vị trí ứng viên mong muốn là gì?"
        }

        total_score = 0
        radar_scores = {}
        st.subheader("Kết quả đánh giá từng mục")

        for name, question in fields.items():
            cv_answer = cv_qa(question)
            jd_question = f"Job description yêu cầu gì về: {name.lower()}?"
            jd_answer = jd_qa(jd_question)
            score = calculate_similarity(cv_answer, jd_answer)
            total_score += score
            radar_scores[name] = score * 100

            with st.expander(f"🔍 {name} (Tương đồng: {score * 100:.0f}%)"):
                st.markdown(f"**Từ CV:** {cv_answer or 'Không tìm thấy'}")
                st.markdown(f"**Từ JD:** {jd_answer or 'Không tìm thấy'}")

        avg_score = total_score / len(fields)
        st.success(f"Mức độ phù hợp tổng thể: **{avg_score * 100:.0f}%**")
        draw_radar(radar_scores)
else:
    st.warning("Vui lòng upload **cả CV** và **JD** để hệ thống phân tích.")
