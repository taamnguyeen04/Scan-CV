from transformers import pipeline
import cv2
import pytesseract
import torch
from collections import defaultdict
from transformers import DistilBertTokenizer, DistilBertForMaskedLM
from pprint import pprint
from icecream import ic
import re

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def cout(a):
    print(a)
    print(type(a))

def process_image_to_dict(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    ocr_data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)

    result_dict = defaultdict(list)

    for i in range(len(ocr_data["text"])):
        if ocr_data["text"][i].strip():
            key = (
                ocr_data["page_num"][i],
                ocr_data["block_num"][i],
                ocr_data["par_num"][i],
                ocr_data["line_num"][i],
            )

            result_dict[key].append(ocr_data["text"][i])

    return dict(result_dict)

def extract_info(text):
    """
    keyword: Education | Experience | Projects | Technical Skills | Soft Skills | Certifications | Awards | Achievements | Languages
    """
    name_match = re.search(r"^[A-Z][a-z]+\s[A-Z][a-z]+(?:\s[A-Z][a-z]+)*", text)

    phone_match = re.search(r"\b\d{9,11}\b", text)

    email_match = re.search(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", text)

    experience = re.search(r"Experience(.*?)(Education | Projects | Technical Skills | Soft Skills | Certifications | Awards | Achievements | Languages |$)", text, re.DOTALL)

    projects = re.search(r"Projects(.*?)(Education | Experience | Technical Skills | Soft Skills | Certifications | Awards | Achievements | Languages |$)", text, re.DOTALL)

    tech_skill = re.search(r"Technical Skills(.*?)(Education | Experience | Projects | Soft Skills | Certifications | Awards | Achievements | Languages |$)", text, re.DOTALL)

    soft_skill =  re.search(r"Soft Skills(.*?)(Education | Experience | Projects | Technical Skills | Certifications | Awards | Achievements | Languages |$)", text, re.DOTALL)

    certifications  =  re.search(r"Certifications(.*?)(Education | Experience | Projects | Technical Skills | Soft Skills | Awards | Achievements | Languages|$)", text, re.DOTALL)

    awards  =  re.search(r"Awards(.*?)(Education | Experience | Projects | Technical Skills | Soft Skills | Certifications | Achievements | Languages|$)", text, re.DOTALL)

    achievements  =  re.search(r"Achievements(.*?)(Education | Experience | Projects | Technical Skills | Soft Skills | Certifications | Awards | Languages|$)", text, re.DOTALL)



    extracted_info = {
        "Name": name_match.group() if name_match else "Not found",
        "Phone": phone_match.group() if phone_match else "Not found",
        "Email": email_match.group() if email_match else "Not found",
        "Skills": tech_skill.group(1).strip() if tech_skill else "Not found",
        "Projects": projects.group(1).strip() if projects else "Not found",
        "Experience": experience.group(1).strip() if experience else "Not found",
        "Soft Skills": soft_skill.group(1).strip() if soft_skill else "Not found",
        "Certifications": certifications.group(1).strip() if certifications else "Not found",
        "Awards": awards.group(1).strip() if awards else "Not found",
        "Achievements": achievements.group(1).strip() if achievements else "Not found",
    }

    return extracted_info

if __name__ == '__main__':
    device = 0 if torch.cuda.is_available() else -1
    fill_mask = pipeline("fill-mask", model="distilbert-base-uncased", device=device)

    result = process_image_to_dict("cv.jpg")
    ic(result.values())
    s = ""
    for k, v in result.items():
        s += ' '.join(v)
        s += ' '
    ic(s)

    info = extract_info(s)
    for key, value in info.items():
        print(f"{key}: {value}\n")


