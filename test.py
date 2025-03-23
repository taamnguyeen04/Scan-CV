import cv2
import pytesseract
import torch
from collections import defaultdict
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from icecream import ic

# Đường dẫn Tesseract OCR
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Load mô hình DeepSeek
MODEL_NAME = "deepseek-ai/DeepSeek-Coder-V2-Base"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, trust_remote_code=True, torch_dtype=torch.bfloat16).cuda()


def process_image_to_text(img_path):
    """Xử lý ảnh CV và trích xuất văn bản bằng Tesseract OCR."""
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ocr_data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)

    text_lines = []
    for i in range(len(ocr_data["text"])):
        if ocr_data["text"][i].strip():
            text_lines.append(ocr_data["text"][i])

    return " ".join(text_lines)


def extract_cv_info(text):
    """Trích xuất thông tin quan trọng từ văn bản bằng DeepSeek."""
    prompt = f"""
    Bạn là một trợ lý AI chuyên phân tích CV. 
    Hãy trích xuất các thông tin quan trọng từ CV sau đây:

    - Họ và tên:
    - Email:
    - Số điện thoại:
    - Học vấn:
    - Kinh nghiệm làm việc:
    - Kỹ năng:

    CV: {text}
    """

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_length=512)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


if __name__ == '__main__':
    # Bước 1: Trích xuất văn bản từ ảnh CV
    text = process_image_to_text("cv.jpg")
    ic(text)

    # Bước 2: Trích xuất thông tin quan trọng
    # extracted_info = extract_cv_info(text)
    # ic(extracted_info)
