import streamlit as st
import os
import base64
import mimetypes
from langchain_community.document_loaders import PDFPlumberLoader, UnstructuredFileLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import LLMChain, RetrievalQA
from langchain_classic.chains.combine_documents.stuff import StuffDocumentsChain
import matplotlib.pyplot as plt
import numpy as np
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
import json
from datetime import datetime
import pandas as pd

load_dotenv()

os.environ["STREAMLIT_WATCH_DISABLE"] = "true"

# ==================== MODELS ====================
def get_gemini_model():
    return ChatGoogleGenerativeAI(
        model="gemini-3-flash-preview",
        temperature=0.1,
    )

def get_openai_model():
    return ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.1,
    )

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="AI CV & JD Analyzer Pro",
    page_icon="logo1.jpg",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== CUSTOM CSS ====================
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    * { font-family: 'Inter', sans-serif; }

    .stApp {
        background: #f6f8fb;
        color: #0f172a;
    }

    .main .block-container {
        max-width: 1200px;
        padding: 1.5rem 2rem 2rem;
        background: #ffffff;
        border-radius: 14px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 8px 24px rgba(15, 23, 42, 0.06);
        margin-top: 1rem;
        margin-bottom: 1rem;
    }

    .app-header {
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1rem 1.25rem;
        background: #ffffff;
        margin-bottom: 1.25rem;
    }

    .app-title {
        margin: 0;
        font-size: 2rem;
        font-weight: 700;
        color: #0f172a;
    }

    .app-subtitle {
        margin-top: 0.35rem;
        margin-bottom: 0;
        color: #475569;
        font-size: 0.98rem;
    }

    .stButton>button {
        background: #1d4ed8;
        color: #fff;
        border: none;
        border-radius: 8px;
        padding: 0.55rem 1rem;
        font-weight: 600;
    }

    .stButton>button:hover {
        background: #1e40af;
    }

    .stFileUploader {
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 0.35rem;
        background: #ffffff;
    }

    .streamlit-expanderHeader {
        background: #f8fafc;
        border-radius: 8px;
        border: 1px solid #e2e8f0;
    }

    .stProgress > div > div > div > div {
        background: #1d4ed8;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        padding: 8px 14px;
        font-weight: 600;
    }

    .stTabs [aria-selected="true"] {
        background: #e2e8f0;
        color: #0f172a;
    }

    .score-badge {
        display: inline-block;
        padding: 7px 14px;
        border-radius: 999px;
        font-weight: 700;
        font-size: 1rem;
        margin: 8px 0;
    }

    .score-excellent { background: #16a34a; color: #fff; }
    .score-good { background: #0284c7; color: #fff; }
    .score-average { background: #ca8a04; color: #fff; }
    .score-poor { background: #dc2626; color: #fff; }

    .suggestion-box {
        background: #fffbeb;
        border: 1px solid #fde68a;
        border-left: 4px solid #f59e0b;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.75rem 0;
    }

    .suggestion-title {
        color: #92400e;
        font-weight: 700;
        font-size: 1.05rem;
        margin-bottom: 0.45rem;
    }

    .suggestion-item {
        color: #334155;
    }
    </style>
""", unsafe_allow_html=True)

# ==================== HEADER ====================
st.markdown('<div class="app-header">', unsafe_allow_html=True)

try:
    st.image("logo2.jpg", width="stretch")
except Exception:
    st.markdown("### 🎯")

st.markdown('<h1 class="app-title">AI CV & JD Analyzer Pro</h1>', unsafe_allow_html=True)
st.markdown('<p class="app-subtitle">Phân tích CV/JD rõ ràng, nhanh và chuyên nghiệp.</p>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# ==================== SIDEBAR ====================
with st.sidebar:
    st.markdown("### ⚙️ Cấu hình phân tích")
    
    analysis_depth = st.select_slider(
        "Độ sâu phân tích",
        options=["Cơ bản", "Tiêu chuẩn", "Chuyên sâu"],
        value="Cơ bản",
        help="Chế độ chuyên sâu sẽ phân tích chi tiết hơn nhưng mất nhiều thời gian hơn"
    )
    
    st.markdown("---")
    st.markdown("### 📊 Các tiêu chí đánh giá")
    
    criteria_weights = {
        "Kỹ năng": st.slider("Kỹ năng", 0, 100, 20, 5),
        "Kinh nghiệm": st.slider("Kinh nghiệm", 0, 100, 20, 5),
        "Học vấn": st.slider("Học vấn", 0, 100, 20, 5),
        "Bằng cấp": st.slider("Bằng cấp/Chứng chỉ", 0, 100, 20, 5),
        "Vị trí": st.slider("Vị trí mong muốn", 0, 100, 20, 5)
    }
    
    total_weight = sum(criteria_weights.values())
    if total_weight != 100:
        st.warning(f"⚠️ Tổng trọng số: {total_weight}% (nên là 100%)")
    else:
        st.success(f"✅ Tổng trọng số: {total_weight}%")
    
    st.markdown("---")
    st.markdown("### 💡 Hướng dẫn nhanh")
    st.caption("• Upload CV và JD\n• Bấm phân tích và xem kết quả\n• Điều chỉnh trọng số nếu cần")

# ==================== MAIN CONTENT ====================
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 📄 Upload CV của ứng viên")
    cv_file = st.file_uploader(
        "Chọn file CV",
        type=["pdf", "docx", "png", "jpg", "jpeg"],
        key="cv",
        help="Hỗ trợ: PDF, Word, hoặc ảnh"
    )
    if cv_file:
        st.success(f"✅ Đã tải: {cv_file.name}")

with col2:
    st.markdown("### 📋 Upload Job Description")
    jd_file = st.file_uploader(
        "Chọn file JD",
        type=["pdf", "docx", "png", "jpg", "jpeg"],
        key="jd",
        help="Hỗ trợ: PDF, Word, hoặc ảnh"
    )
    if jd_file:
        st.success(f"✅ Đã tải: {jd_file.name}")

# ==================== SIMILARITY MODEL ====================
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

# ==================== VISUALIZATION ====================
def draw_enhanced_radar(scores_dict):
    """Vẽ biểu đồ radar với thiết kế chuyên nghiệp hơn"""
    labels = list(scores_dict.keys())
    values = list(scores_dict.values())
    
    # Tạo dữ liệu cho radar chart
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    values += values[:1]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    
    # Vẽ đường và fill
    ax.plot(angles, values, 'o-', linewidth=3, color='#667eea', markersize=8)
    ax.fill(angles, values, alpha=0.25, color='#764ba2')
    
    # Styling
    ax.set_thetagrids(np.degrees(angles[:-1]), labels, fontsize=12, fontweight='bold')
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(['20%', '40%', '60%', '80%', '100%'], fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Title
    ax.set_title("Biểu đồ phân tích mức độ phù hợp", 
                 fontsize=16, fontweight='bold', pad=20)
    
    # Background color
    ax.set_facecolor('#f8f9fa')
    fig.patch.set_facecolor('white')
    
    return fig

def get_score_badge(score):
    """Trả về HTML badge dựa trên điểm số"""
    if score >= 80:
        return f'<span class="score-badge score-excellent">Xuất sắc: {score:.0f}%</span>'
    elif score >= 65:
        return f'<span class="score-badge score-good">Tốt: {score:.0f}%</span>'
    elif score >= 50:
        return f'<span class="score-badge score-average">Trung bình: {score:.0f}%</span>'
    else:
        return f'<span class="score-badge score-poor">Cần cải thiện: {score:.0f}%</span>'

# ==================== PROMPTS ====================
system_prompt = PromptTemplate.from_template("""
Bạn là một chuyên gia tuyển dụng hàng đầu với hơn 15 năm kinh nghiệm trong ngành nhân sự.
Nhiệm vụ của bạn là phân tích hồ sơ ứng viên (CV) và bản mô tả công việc (JD) một cách khách quan và chi tiết.

NGUYÊN TẮC PHÂN TÍCH:
- Chỉ sử dụng thông tin có trong context
- Đưa ra đánh giá cụ thể, có dẫn chứng
- Không suy đoán hoặc thêm thông tin không có
- Nếu thiếu thông tin, hãy nói rõ "Không đủ thông tin để đánh giá"

Context: {context}
Question: {question}

Hãy trả lời một cách chi tiết, chuyên nghiệp và có cấu trúc:
""")

suggestion_prompt = PromptTemplate.from_template("""
Bạn là một chuyên gia tư vấn nghề nghiệp, giúp ứng viên cải thiện CV để phù hợp hơn với công việc mong muốn.

Dựa trên phân tích so sánh giữa CV và JD, hãy đưa ra 5-7 gợi ý cải thiện CỤ THỂ và KHẢ THI.

THÔNG TIN PHÂN TÍCH:
CV của ứng viên: {cv_summary}
Job Description yêu cầu: {jd_summary}
Các điểm yếu đã phát hiện: {weaknesses}

YÊU CẦU ĐỀ XUẤT:
1. Mỗi gợi ý phải CỤ THỂ, có thể thực hiện ngay
2. Ưu tiên những thay đổi có tác động lớn nhất
3. Tránh những lời khuyên chung chung, mơ hồ
4. Đưa ra ví dụ minh họa khi có thể

Hãy trả lời dưới dạng JSON với format:
{{
    "suggestions": [
        {{"priority": "Cao", "category": "Kỹ năng", "title": "Tiêu đề ngắn gọn", "detail": "Mô tả chi tiết cách thực hiện"}},
        ...
    ]
}}

Chỉ trả về JSON, không thêm text khác:
""")

# ==================== QA BUILDER ====================
@st.cache_resource
def build_qa_chain(uploaded_file):
    file_name = "temp_" + uploaded_file.name
    with open(file_name, "wb") as f:
        f.write(uploaded_file.getvalue())
    
    ext = uploaded_file.name.split(".")[-1].lower()
    
    # Xử lý file ảnh bằng vision model (không cần OCR/Tesseract)
    if ext in ["jpg", "jpeg", "png"]:
        mime_type, _ = mimetypes.guess_type(file_name)
        if not mime_type:
            mime_type = "image/jpeg"

        with open(file_name, "rb") as img_file:
            image_base64 = base64.b64encode(img_file.read()).decode("utf-8")

        def qa_fn(question):
            instruction = (
                "Bạn là chuyên gia tuyển dụng. Chỉ dùng nội dung nhìn thấy trong ảnh CV/JD để trả lời. "
                "Nếu ảnh thiếu thông tin, hãy nói rõ không đủ thông tin."
            )
            message = HumanMessage(
                content=[
                    {"type": "text", "text": f"{instruction}\n\nCâu hỏi: {question}"},
                    {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{image_base64}"}},
                ]
            )

            try:
                llm = get_openai_model()
                response = llm.invoke([message])
            except Exception:
                st.info("🔄 Chuyển sang Gemini model...")
                llm = get_gemini_model()
                response = llm.invoke([message])

            return response.content if hasattr(response, "content") else str(response)

        return qa_fn
    
    # Xử lý PDF/DOCX với chunking thông minh
    if ext == "pdf":
        loader = PDFPlumberLoader(file_name)
    else:
        loader = UnstructuredFileLoader(file_name)
    
    docs = loader.load()
    
    embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
    splitter = SemanticChunker(embedder)
    chunks = splitter.split_documents(docs)
    
    vector = FAISS.from_documents(chunks, embedder)
    retriever = vector.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    
    def create_qa_chain_with_model(model):
        llm_chain = LLMChain(llm=model, prompt=system_prompt)
        combine_docs_chain = StuffDocumentsChain(
            llm_chain=llm_chain,
            document_variable_name="context",
            document_prompt=PromptTemplate(
                input_variables=["page_content", "source"],
                template="Context:\ncontent:{page_content}\nsource:{source}"
            )
        )
        return RetrievalQA(combine_documents_chain=combine_docs_chain, retriever=retriever)
    
    gemini_qa_chain = create_qa_chain_with_model(get_gemini_model())
    openai_qa_chain = create_qa_chain_with_model(get_openai_model())
    
    def qa_fn(question):
        try:
            return openai_qa_chain.run(question).strip()
        except Exception as e:
            st.info("🔄 Chuyển sang Gemini model...")
            return gemini_qa_chain.run(question).strip()
    
    return qa_fn

# ==================== SUGGESTION GENERATOR ====================
def generate_suggestions(cv_qa, jd_qa, field_results):
    """Tạo gợi ý cải thiện CV dựa trên kết quả phân tích"""
    
    # Tổng hợp thông tin
    cv_summary = "\n".join([f"- {k}: {v['cv']}" for k, v in field_results.items()])
    jd_summary = "\n".join([f"- {k}: {v['jd']}" for k, v in field_results.items()])
    
    # Tìm điểm yếu (score < 60%)
    weaknesses = [k for k, v in field_results.items() if v['score'] < 0.6]
    weakness_text = ", ".join(weaknesses) if weaknesses else "Không có điểm yếu rõ ràng"
    
    # Tạo prompt
    prompt = suggestion_prompt.format(
        cv_summary=cv_summary,
        jd_summary=jd_summary,
        weaknesses=weakness_text
    )
    
    try:
        llm = get_openai_model()
        response = llm.invoke(prompt)
        content = response.content.strip()
        
        # Xử lý JSON response
        if content.startswith("```json"):
            content = content[7:]
        if content.endswith("```"):
            content = content[:-3]
        
        suggestions_data = json.loads(content.strip())
        return suggestions_data.get("suggestions", [])
    
    except Exception as e:
        st.error(f"Lỗi khi tạo gợi ý: {str(e)}")
        return []

# ==================== MAIN ANALYSIS ====================
if cv_file and jd_file:
    
    # Progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    with st.spinner(""):
        status_text.text("🔍 Đang tải và xử lý tài liệu...")
        progress_bar.progress(10)
        
        cv_qa = build_qa_chain(cv_file)
        progress_bar.progress(30)
        
        jd_qa = build_qa_chain(jd_file)
        progress_bar.progress(50)
        
        status_text.text("🧠 Đang phân tích và so sánh...")
        
        # Định nghĩa các trường phân tích
        fields = {
            "Kỹ năng": {
                "cv_question": "Liệt kê CHI TIẾT tất cả kỹ năng chuyên môn, kỹ năng mềm và công cụ mà ứng viên có.",
                "jd_question": "Job description yêu cầu những kỹ năng gì? Liệt kê cụ thể các kỹ năng bắt buộc và ưu tiên."
            },
            "Kinh nghiệm": {
                "cv_question": "Mô tả CHI TIẾT kinh nghiệm làm việc của ứng viên: số năm, vị trí, công ty, dự án và thành tích.",
                "jd_question": "Job description yêu cầu kinh nghiệm như thế nào? Số năm? Lĩnh vực? Vị trí tương tự?"
            },
            "Học vấn": {
                "cv_question": "Trình độ học vấn của ứng viên: bằng cấp, trường, chuyên ngành, năm tốt nghiệp, GPA nếu có.",
                "jd_question": "Job description yêu cầu trình độ học vấn tối thiểu là gì? Có yêu cầu chuyên ngành cụ thể không?"
            },
            "Bằng cấp": {
                "cv_question": "Liệt kê tất cả bằng cấp, chứng chỉ chuyên môn, chứng chỉ kỹ năng mà ứng viên có.",
                "jd_question": "Job description có yêu cầu chứng chỉ, bằng cấp chuyên môn nào không?"
            },
            "Vị trí": {
                "cv_question": "Vị trí công việc mà ứng viên đang tìm kiếm hoặc mục tiêu nghề nghiệp là gì?",
                "jd_question": "Vị trí tuyển dụng và vai trò công việc cụ thể là gì?"
            }
        }
        
        # Phân tích từng trường
        field_results = {}
        progress_step = 40 / len(fields)
        current_progress = 50
        
        for name, questions in fields.items():
            cv_answer = cv_qa(questions["cv_question"])
            jd_answer = jd_qa(questions["jd_question"])
            score = calculate_similarity(cv_answer, jd_answer)
            
            field_results[name] = {
                "cv": cv_answer,
                "jd": jd_answer,
                "score": score
            }
            
            current_progress += progress_step
            progress_bar.progress(int(current_progress))
        
        status_text.text("💡 Đang tạo gợi ý cải thiện...")
        progress_bar.progress(90)
        
        # Tạo gợi ý
        suggestions = generate_suggestions(cv_qa, jd_qa, field_results)
        
        progress_bar.progress(100)
        status_text.text("✅ Hoàn tất phân tích!")
        
    # Xóa progress bar sau 1 giây
    import time
    time.sleep(1)
    progress_bar.empty()
    status_text.empty()
    
    # ==================== RESULTS DISPLAY ====================
    st.markdown("---")
    
    # Tính điểm tổng thể (có trọng số)
    total_weighted_score = sum(
        field_results[name]["score"] * (criteria_weights[name] / 100) 
        for name in field_results.keys()
    )
    avg_score = total_weighted_score * 100
    
    # Hiển thị điểm tổng thể
    st.markdown("## 📊 Kết quả phân tích tổng quan")
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.markdown(get_score_badge(avg_score), unsafe_allow_html=True)
        st.markdown(f"### Mức độ phù hợp: **{avg_score:.1f}%**")
        
        # Đánh giá tổng quan
        if avg_score >= 80:
            st.success("🌟 **Ứng viên rất phù hợp** với vị trí này. Hồ sơ thể hiện sự tương đồng cao với yêu cầu công việc.")
        elif avg_score >= 65:
            st.info("👍 **Ứng viên phù hợp** với vị trí. Có một số điểm cần cải thiện để tăng cơ hội.")
        elif avg_score >= 50:
            st.warning("⚠️ **Ứng viên tương đối phù hợp**. Cần cải thiện một số kỹ năng/kinh nghiệm chính.")
        else:
            st.error("❌ **Mức độ phù hợp thấp**. CV và JD có nhiều điểm khác biệt đáng kể.")
    
    with col2:
        st.markdown("#### 📈 Điểm cao nhất")
        best_field = max(field_results.items(), key=lambda x: x[1]["score"])
        st.metric(best_field[0], f"{best_field[1]['score']*100:.0f}%", delta="Điểm mạnh")
    
    with col3:
        st.markdown("#### 📉 Cần cải thiện")
        worst_field = min(field_results.items(), key=lambda x: x[1]["score"])
        st.metric(worst_field[0], f"{worst_field[1]['score']*100:.0f}%", delta="Ưu tiên")
    
    # Tabs cho các phần khác nhau
    tab1, tab2, tab3 = st.tabs(["📋 Chi tiết từng mục", "📊 Biểu đồ trực quan", "💡 Gợi ý cải thiện"])
    
    with tab1:
        st.markdown("### Phân tích chi tiết theo từng tiêu chí")
        
        for name, result in field_results.items():
            score_pct = result["score"] * 100
            weight = criteria_weights[name]
            
            with st.expander(f"🔍 {name} - **{score_pct:.0f}%** (Trọng số: {weight}%)", expanded=False):
                col_a, col_b = st.columns(2)
                
                with col_a:
                    st.markdown("##### 📄 Từ CV của ứng viên")
                    st.info(result["cv"] if result["cv"] else "Không tìm thấy thông tin")
                
                with col_b:
                    st.markdown("##### 📋 Yêu cầu từ JD")
                    st.warning(result["jd"] if result["jd"] else "Không có yêu cầu cụ thể")
                
                # Progress bar cho điểm
                st.progress(result["score"])
                
                # Nhận xét
                if score_pct >= 80:
                    st.success(f"✅ **Xuất sắc** - Ứng viên đáp ứng rất tốt yêu cầu về {name.lower()}")
                elif score_pct >= 65:
                    st.info(f"👍 **Tốt** - Ứng viên có {name.lower()} phù hợp với yêu cầu")
                elif score_pct >= 50:
                    st.warning(f"⚠️ **Trung bình** - {name} của ứng viên cần bổ sung thêm")
                else:
                    st.error(f"❌ **Cần cải thiện** - Có sự chênh lệch đáng kể về {name.lower()}")
    
    with tab2:
        st.markdown("### Biểu đồ phân tích đa chiều")
        
        # Chuẩn bị dữ liệu cho radar chart
        radar_scores = {name: result["score"] * 100 for name, result in field_results.items()}
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig = draw_enhanced_radar(radar_scores)
            st.pyplot(fig)
        
        with col2:
            st.markdown("#### 📊 Bảng điểm chi tiết")
            
            df = pd.DataFrame([
                {
                    "Tiêu chí": name,
                    "Điểm": f"{result['score']*100:.1f}%",
                    "Trọng số": f"{criteria_weights[name]}%",
                    "Đóng góp": f"{result['score']*criteria_weights[name]:.1f}"
                }
                for name, result in field_results.items()
            ])
            
            st.dataframe(df, use_container_width=True, hide_index=True)
            
            st.markdown(f"**Tổng điểm có trọng số:** {avg_score:.1f}%")
    
    with tab3:
        st.markdown("### 💡 Gợi ý cải thiện CV")
        
        if suggestions:
            st.info("Dựa trên phân tích AI, đây là các gợi ý giúp tăng mức độ phù hợp của CV với JD:")
            
            # Phân loại theo priority
            high_priority = [s for s in suggestions if s.get("priority") == "Cao"]
            medium_priority = [s for s in suggestions if s.get("priority") == "Trung bình"]
            low_priority = [s for s in suggestions if s.get("priority") == "Thấp"]
            
            for priority_name, priority_list, icon in [
                ("🔴 Ưu tiên cao", high_priority, "🔴"),
                ("🟡 Ưu tiên trung bình", medium_priority, "🟡"),
                ("🟢 Ưu tiên thấp", low_priority, "🟢")
            ]:
                if priority_list:
                    st.markdown(f"#### {priority_name}")
                    
                    for idx, suggestion in enumerate(priority_list, 1):
                        with st.container():
                            st.markdown(f"""
                            <div class="suggestion-box">
                                <div class="suggestion-title">{icon} {suggestion.get('category', 'Chung')}: {suggestion.get('title', 'Gợi ý')}</div>
                                <div class="suggestion-item">{suggestion.get('detail', 'Chi tiết không có sẵn')}</div>
                            </div>
                            """, unsafe_allow_html=True)
        else:
            st.warning("Không thể tạo gợi ý tự động. Vui lòng xem phần phân tích chi tiết.")
            
            # Gợi ý thủ công dựa trên điểm thấp
            st.markdown("#### 📝 Gợi ý chung")
            low_score_fields = [name for name, result in field_results.items() if result["score"] < 0.6]
            
            if low_score_fields:
                for field in low_score_fields:
                    st.markdown(f"- **{field}**: Cần bổ sung hoặc làm nổi bật hơn phần này trong CV để phù hợp với yêu cầu JD")
    
    # Footer với timestamp
    st.markdown("---")
    st.markdown(f"*Phân tích được thực hiện lúc: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
    st.markdown("*Kết quả chỉ mang tính chất tham khảo. Quyết định cuối cùng thuộc về nhà tuyển dụng.*")

else:
    # Hướng dẫn khi chưa upload file
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("""
        ### 📄 Bước 1: Upload CV
        - Định dạng: PDF/Word/Ảnh
        - Nội dung rõ, dễ đọc
        """)

    with col2:
        st.info("""
        ### 📋 Bước 2: Upload JD
        - Định dạng: PDF/Word/Ảnh
        - Nên có yêu cầu chi tiết
        """)
    
    st.warning("⚠️ Vui lòng upload **cả CV và JD** để bắt đầu phân tích")
    
    # Sample data hoặc demo
    with st.expander("💡 Tính năng nổi bật của hệ thống"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            #### 🎯 Phân tích thông minh
            - AI phân tích sâu
            - So sánh đa chiều
            - Đánh giá khách quan
            """)
        
        with col2:
            st.markdown("""
            #### 📊 Trực quan hóa
            - Biểu đồ radar
            - Bảng điểm chi tiết
            - Giao diện trực quan
            """)
        
        with col3:
            st.markdown("""
            #### 💡 Gợi ý cải thiện
            - AI đề xuất tự động
            - Hướng dẫn cụ thể
            - Ưu tiên hợp lý
            """)