import streamlit as st
from qa_chain import build_qa_chain, extract_info
from similarity import calculate_similarity, draw_radar
from PIL import Image

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

cv_file = st.file_uploader("Upload CV (PDF)", type="pdf", key="cv")
jd_file = st.file_uploader("Upload Job Description (PDF)", type="pdf", key="jd")

if cv_file and jd_file:
    with st.spinner("\U0001F50D Đang phân tích CV và JD..."):
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
            cv_answer = extract_info(cv_qa, question)
            jd_question = f"Job description yêu cầu gì về: {name.lower()}?"
            jd_answer = extract_info(jd_qa, jd_question)
            score = calculate_similarity(cv_answer, jd_answer)
            total_score += score
            radar_scores[name] = score * 100

            with st.expander(f"\U0001F50D {name} (Tương đồng: {score * 100:.0f}%)"):
                st.markdown(f"**Từ CV:** {cv_answer or 'Không tìm thấy'}")
                st.markdown(f"**Từ JD:** {jd_answer or 'Không tìm thấy'}")

        avg_score = total_score / len(fields)
        st.success(f"Mức độ phù hợp tổng thể: **{avg_score * 100:.0f}%**")
        draw_radar(radar_scores)
else:
    st.warning("Vui lòng upload **cả CV** và **JD** để hệ thống phân tích.")