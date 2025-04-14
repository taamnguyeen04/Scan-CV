# 📄 CV & JD Matching System

This is an AI-powered web application that **analyzes and evaluates the compatibility between a candidate's CV and a Job Description (JD)**. It leverages advanced NLP techniques and Large Language Models (LLMs) to extract and compare key information.

## 🚀 Key Features

- ✅ Upload CV and JD in PDF format  
- 🤖 Automatically extract key details such as: skills, education, experience, certifications, and desired position  
- 🔍 Evaluate how well a candidate’s profile matches the job requirements  
- 📊 Visualize similarity scores using a radar chart  
- 🧠 Utilize powerful LLM (via Ollama) for deep contextual understanding  

## 🛠️ Technologies Used

| Component             | Library / Technology                                                            |
|-----------------------|----------------------------------------------------------------------------------|
| User Interface        | `Streamlit`                                                                     |
| PDF Parsing           | `PDFPlumberLoader` from `langchain_community`                                   |
| Semantic Splitting    | `SemanticChunker` from `langchain_experimental`                                 |
| Text Embeddings       | `HuggingFaceEmbeddings`, `SentenceTransformer` (`all-MiniLM-L6-v2`)             |
| Vector Database       | `FAISS`                                                                          |
| LLM                   | `Ollama` using `deepseek-coder-v2:16b`                                           |
| Prompt Chains         | `RetrievalQA`, `LLMChain`, `PromptTemplate`, `StuffDocumentsChain` (LangChain)  |
| Similarity Scoring    | Cosine similarity with `SentenceTransformer`                                     |
| Visualization         | `matplotlib`                                                                     |

## 📈 How It Works

1. **Upload CV and JD PDF files**  
2. **Preprocess and split documents into semantic chunks**  
3. **Create vector embeddings and use FAISS for semantic search**  
4. **Extract answers to key questions using an LLM**  
5. **Calculate similarity scores for each category**  
6. **Display a summary score and a radar chart for detailed comparison**  

## 📦 Installation

```bash
pip install -r requirements.txt
```

> 💡 Make sure you have [Ollama](https://ollama.com) installed and the model `deepseek-coder-v2:16b` downloaded.

## 🚀 Run the App

```bash
streamlit run app.py
```

## 🧩 Future Enhancements

- Add multilingual support  
- Allow manual input of CV/JD text  
- Add more granular sub-criteria scoring  
- Generate improvement suggestions for candidates  
