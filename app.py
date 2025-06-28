# app.py
import streamlit as st
import os
import tempfile
import pytesseract
from pdf2image import convert_from_bytes
from PIL import Image
import google.generativeai as genai
import pandas as pd
import json, re
import cv2
import numpy as np
from PyPDF2 import PdfReader
from pdf2image import convert_from_path, convert_from_bytes

# Setup session state for API key
if "api_key" not in st.session_state:
    st.session_state.api_key = ""

# Streamlit UI
st.set_page_config(page_title="Menu Formatter", layout="centered")
st.title("📋 Restaurant Menu Formatter")
st.write("Upload your restaurant menu (PDF/Image/Excel/CSV) and get structured Excel output!")

# Gemini API Key Input
st.session_state.api_key = st.text_input("🔑 Enter your Gemini API Key", type="password", value=st.session_state.api_key)
if not st.session_state.api_key:
    st.warning("Please enter your Gemini API key to continue.")
    st.stop()

genai.configure(api_key=st.session_state.api_key)

# File upload
uploaded_file = st.file_uploader("📂 Upload Menu File", type=["pdf", "jpg", "jpeg", "png", "xlsx", "csv"])
method = st.radio("Processing Method", ["OCR (Tesseract)", "Direct to Gemini"], index=0)
output_name = st.text_input("📁 Output Excel File Name", value="formatted_menu.xlsx")

# Preprocess image for better OCR
def preprocess_image(pil_img):
    img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2GRAY)
    _, bin_img = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return Image.fromarray(bin_img)

# Extract OCR text
def extract_text(file_bytes, ext):
    all_text = ""
    if ext == ".pdf":
        images = convert_from_path(file_bytes)
        for idx, img in enumerate(images):
            clean_img = preprocess_image(img)
            text = pytesseract.image_to_string(clean_img, config="--psm 4")
            all_text += text + "\n"
    elif ext in [".jpg", ".jpeg", ".png"]:
        img = Image.open(file_bytes)
        clean_img = preprocess_image(img)
        all_text = pytesseract.image_to_string(clean_img, config="--psm 4")
    elif ext in [".xlsx", ".csv"]:
        df = pd.read_excel(file_bytes) if ext == ".xlsx" else pd.read_csv(file_bytes)
        all_text = "\n".join(df.astype(str).fillna("").agg(" | ".join, axis=1))
    return all_text.strip()

# Extract direct text
def extract_raw_text(file_bytes, ext):
    if ext in [".xlsx", ".csv"]:
        df = pd.read_excel(file_bytes) if ext == ".xlsx" else pd.read_csv(file_bytes)
        return "\n".join(df.astype(str).fillna("").agg(" | ".join, axis=1))
    elif ext == ".pdf":
        reader = PdfReader(file_bytes)
        return "\n".join([page.extract_text() or "" for page in reader.pages])
    return ""

# Ask Gemini
def ask_gemini(prompt_text):
    model = genai.GenerativeModel("gemini-1.5-flash")
    prompt = f"""
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
"""
    response = model.generate_content(prompt)
    return response.text

# Parse and save
def parse_json_response(text, output_path):
    text = re.sub(r"```+", "", text)
    text = re.sub(r",\s*(\}|\])", r"\1", text)
    text = text.replace("\\n", " ").replace("\\", "")
    match = re.search(r"\[\s*{.*", text, re.DOTALL)
    if not match:
        st.error("❌ Gemini response doesn't look like valid JSON.")
        return
    cleaned = match.group()

    open_brackets = cleaned.count("[")
    close_brackets = cleaned.count("]")
    open_braces = cleaned.count("{")
    close_braces = cleaned.count("}")

    cleaned += "}" * (open_braces - close_braces)
    cleaned += "]" * (open_brackets - close_brackets)

    try:
        parsed = json.loads(cleaned)
        symbols_to_remove = r"[{}\[\]\"']"
        for row in parsed:
            for key, value in row.items():
                if isinstance(value, str):
                    row[key] = re.sub(symbols_to_remove, "", value).strip()
        df = pd.DataFrame(parsed)
        df.to_excel(output_path, index=False)
        st.success("✅ Menu formatted and saved successfully.")
        st.download_button("📥 Download Excel", data=open(output_path, "rb").read(), file_name=output_name)
    except Exception as e:
        st.error("❌ JSON Parsing Error")
        st.error(str(e))

# Process
if st.button("🚀 Run Formatter"):
    if uploaded_file is None:
        st.warning("Please upload a file first.")
    else:
        ext = os.path.splitext(uploaded_file.name)[1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name
        extracted = extract_text(tmp_path, ext) if "OCR" in method else extract_raw_text(tmp_path, ext)
        if not extracted:
            st.error("❌ Failed to extract any text.")
        else:
            gemini_response = ask_gemini(extracted)
            output_path = os.path.join(tempfile.gettempdir(), output_name)
            parse_json_response(gemini_response, output_path)