from sentence_transformers import SentenceTransformer, util
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

similarity_model = SentenceTransformer("all-MiniLM-L6-v2")

def calculate_similarity(text1, text2):
    if not text1 or not text2:
        return 0.0
    emb1 = similarity_model.encode(text1, convert_to_tensor=True)
    emb2 = similarity_model.encode(text2, convert_to_tensor=True)
    return round(util.pytorch_cos_sim(emb1, emb2).item(), 2)

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