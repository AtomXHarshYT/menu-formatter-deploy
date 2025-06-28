import os
import re
import json
import tempfile
import logging
import pandas as pd
import numpy as np
import cv2
import pytesseract
from PIL import Image
from pdf2image import convert_from_bytes
from PyPDF2 import PdfReader
import google.generativeai as genai
import streamlit as st

# -------------------- Configuration --------------------
TESSERACT_PATH = "/usr/bin/tesseract"  # Update if needed
POPPLER_PATH = "/usr/bin"              # Render's Poppler location (auto works on Render)
# -------------------------------------------------------

pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')


def preprocess_image(pil_img):
    img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2GRAY)
    _, bin_img = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return Image.fromarray(bin_img)


def extract_text(file_bytes, ext):
    all_text = ""
    if ext == '.pdf':
        images = convert_from_bytes(file_bytes, poppler_path=POPPLER_PATH)
        for idx, img in enumerate(images):
            st.info(f"OCR on page {idx + 1}")
            clean_img = preprocess_image(img)
            text = pytesseract.image_to_string(clean_img, config='--psm 4')
            all_text += text + "\n"
    elif ext in ['.jpg', '.jpeg', '.png']:
        img = Image.open(tempfile.NamedTemporaryFile(delete=False))
        img.write(file_bytes)
        img = Image.open(img.name)
        clean_img = preprocess_image(img)
        all_text = pytesseract.image_to_string(clean_img, config='--psm 4')
    elif ext in ['.xlsx', '.csv']:
        df = pd.read_excel(file_bytes) if ext == '.xlsx' else pd.read_csv(file_bytes)
        all_text = "\n".join(df.astype(str).fillna('').agg(' | '.join, axis=1))
    else:
        st.error("Unsupported file type.")
    return all_text.strip()


def extract_raw_text(file_bytes, ext):
    if ext in ['.xlsx', '.csv']:
        df = pd.read_excel(file_bytes) if ext == '.xlsx' else pd.read_csv(file_bytes)
        return "\n".join(df.astype(str).fillna('').agg(' | '.join, axis=1))
    elif ext == '.pdf':
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(file_bytes)
                tmp.seek(0)
                reader = PdfReader(tmp.name)
                return "\n".join(page.extract_text() or "" for page in reader.pages)
        except:
            st.warning("PDF might be image-based. Consider using OCR mode.")
            return ""
    elif ext in ['.jpg', '.jpeg', '.png']:
        return ""
    else:
        return ""


def ask_gemini(api_key, prompt_text):
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-1.5-flash")
    prompt = f'''
You are a professional menu parser. Convert the following text into a JSON array where each row is flat and independent.

Each item must include:
- "Category"
- "Item Name"
- "Description" (optional)
- "Variation Type" (or blank)
- "Price"

If an item has multiple variations (e.g. Half and Full), split them into separate rows (with same Category and Item Name). Avoid nested arrays. Output flat rows only.

Recognize these keywords as variations: half, full, small, medium, large, regular, mini, jumbo, single, double, combo, plate, bowl, glass, bottle, cup, pack, box, piece, slice, scoop, shot, bucket, 100g, 250g, 500g, 1kg, 2kg, 100ml, 250ml, 500ml, 1l, 2l, hot, cold, iced, dry, gravy, fried, tandoori, steamed, veg, egg, chicken, mutton, fish, prawns, paneer, mushroom, soya, jain, spicy, non-spicy, extra spicy, with cheese, extra cheese, no cheese, with butter, extra butter, no onion, no garlic, combo, thali, meal, family pack, party pack, takeaway, parcel, dine-in, delivery.

Return only the JSON array. Do not explain.

Menu text:
{prompt_text}
'''
    response = model.generate_content(prompt)
    return response.text


def parse_json_response(text):
    text = text.strip()
    text = re.sub(r'```+', '', text)
    text = re.sub(r',\s*(\}|\])', r'\1', text)
    text = text.replace('\\n', ' ').replace('\\', '')

    json_block = re.search(r'\[\s*{.*', text, re.DOTALL)
    if not json_block:
        raise ValueError("No JSON array found.")
    cleaned = json_block.group()

    open_brackets = cleaned.count("[")
    close_brackets = cleaned.count("]")
    open_braces = cleaned.count("{")
    close_braces = cleaned.count("}")

    cleaned += "}" * (open_braces - close_braces)
    cleaned += "]" * (open_brackets - close_braces)

    parsed = json.loads(cleaned)

    symbols_to_remove = r"[{}\[\]\"']"
    for row in parsed:
        for key, value in row.items():
            if isinstance(value, str):
                row[key] = re.sub(symbols_to_remove, '', value).strip()

    return pd.DataFrame(parsed)


# -------------------- Streamlit UI --------------------

st.set_page_config(page_title="Menu Formatter", layout="centered")

st.title("📋 AI Menu Formatter")
st.caption("Upload your restaurant menu (PDF/Image/Excel), choose a method, and get a clean structured Excel menu!")

api_key = st.text_input("🔑 Enter your Gemini API Key", type="password")
file = st.file_uploader("📂 Upload Menu File", type=["pdf", "jpg", "jpeg", "png", "xlsx", "csv"])
method = st.radio("⚙️ Select Method", ["OCR using Tesseract", "Direct Text (PDF/Excel Only)"])

if st.button("Process Menu"):
    if not api_key:
        st.warning("Please enter your Gemini API key.")
    elif not file:
        st.warning("Please upload a menu file.")
    else:
        ext = os.path.splitext(file.name)[1].lower()
        bytes_data = file.read()
        extracted_text = extract_text(bytes_data, ext) if "OCR" in method else extract_raw_text(bytes_data, ext)

        if not extracted_text:
            st.error("No text extracted.")
        else:
            st.text_area("📝 Extracted Text", extracted_text, height=200)
            st.info("Sending to Gemini...")
            gemini_output = ask_gemini(api_key, extracted_text)

            try:
                df = parse_json_response(gemini_output)
                st.success("✅ Menu parsed successfully!")
                st.dataframe(df)

                st.download_button(
                    label="📥 Download Excel",
                    data=df.to_excel(index=False, engine='openpyxl'),
                    file_name="formatted_menu.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            except Exception as e:
                st.error(f"JSON parsing failed: {e}")