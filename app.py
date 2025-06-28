import os
import pytesseract
from pdf2image import convert_from_path
from PIL import Image
import google.generativeai as genai
import pandas as pd
import logging
import json, re
import cv2
import numpy as np
from tkinter import Tk, filedialog, messagebox

# -------------------- CONFIGURATION --------------------
TESSERACT_PATH = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
POPPLER_PATH = r'C:\poppler-24.08.0\Library\bin'
GENAI_API_KEY = 'AIzaSyBQKUZnZrH4guk0DWaaPqtly-oRixR7nCc'  # Replace with your Gemini API Key
# -------------------------------------------------------

# Setup
pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH
genai.configure(api_key=GENAI_API_KEY)
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

API_KEY_FILE = os.path.expanduser("~/.gemini_api_key.txt")

if os.path.exists(API_KEY_FILE):
    with open(API_KEY_FILE, "r") as f:
        GENAI_API_KEY = f.read().strip()
else:
    from tkinter.simpledialog import askstring
    GENAI_API_KEY = askstring("Gemini API Key", "Enter your Gemini API key:")
    with open(API_KEY_FILE, "w") as f:
        f.write(GENAI_API_KEY)

def preprocess_image(pil_img):
    img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2GRAY)
    _, bin_img = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return Image.fromarray(bin_img)


def extract_text(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    all_text = ""
    if ext == '.pdf':
        images = convert_from_path(file_path, poppler_path=POPPLER_PATH)
        for idx, img in enumerate(images):
            logging.info(f"OCR on page {idx + 1}...")
            clean_img = preprocess_image(img)
            text = pytesseract.image_to_string(clean_img, config='--psm 4')
            all_text += text + "\n"
    elif ext in ['.jpg', '.jpeg', '.png']:
        img = Image.open(file_path)
        clean_img = preprocess_image(img)
        all_text = pytesseract.image_to_string(clean_img, config='--psm 4')
    elif ext in ['.xlsx', '.csv']:
        df = pd.read_excel(file_path) if ext == '.xlsx' else pd.read_csv(file_path)
        all_text = "\n".join(df.astype(str).fillna('').agg(' | '.join, axis=1))
    else:
        logging.error("Unsupported file type.")
    with open("ocr_output.txt", "w", encoding="utf-8") as f:
        f.write(all_text)
    return all_text.strip()


def extract_raw_text(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    if ext in ['.xlsx', '.csv']:
        df = pd.read_excel(file_path) if ext == '.xlsx' else pd.read_csv(file_path)
        return "\n".join(df.astype(str).fillna('').agg(' | '.join, axis=1))
    elif ext == '.pdf':
        try:
            from PyPDF2 import PdfReader
            reader = PdfReader(file_path)
            return "\n".join(page.extract_text() or "" for page in reader.pages)
        except Exception as e:
            logging.warning("PDF might be image-based. Consider using OCR mode.")
            return ""
    elif ext in ['.jpg', '.jpeg', '.png']:
        return ""  # No direct text from images
    else:
        return ""


def ask_gemini(prompt_text):
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


def parse_json_response(text, output_file):
    try:
        with open("raw_gemini_output.txt", "w", encoding="utf-8") as f:
            f.write(text)

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

        with open("debug_cleaned_json.txt", "w", encoding="utf-8") as f:
            f.write(cleaned)

        print("\n✅ Gemini response saved:")
        print("   → raw_gemini_output.txt")
        print("   → debug_cleaned_json.txt")
        input("🟡 Clean the JSON if needed and press ENTER to continue...")

        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError as e:
            logging.error(f"🔴 JSON parsing failed: {e}")
            logging.error("Check debug_cleaned_json.txt and try again.")
            return

        # Clean unwanted characters from each string field
        symbols_to_remove = r"[{}\[\]\"']"
        for row in parsed:
            for key, value in row.items():
                if isinstance(value, str):
                    row[key] = re.sub(symbols_to_remove, '', value).strip()

        df = pd.DataFrame(parsed)
        df.to_excel(output_file, index=False)
        logging.info(f"✅ Final structured data saved to: {output_file}")

    except Exception as e:
        logging.error("❌ Unexpected error during parsing or saving.")
        logging.error(str(e))


def main():
    logging.info("🔥 Starting Menu Formatter with Method Selection")

    root = Tk()
    root.withdraw()

    input_path = filedialog.askopenfilename(
        title="Select menu file (PDF/Image/Excel/CSV)",
        filetypes=[("All Supported", "*.pdf *.jpg *.jpeg *.png *.xlsx *.csv")]
    )
    if not input_path:
        logging.warning("No file selected. Exiting.")
        return

    method = messagebox.askyesno(
        "Select Processing Method",
        "Yes = Use OCR (Tesseract for clean scan)\nNo = Directly send file to Gemini"
    )

    save_path = filedialog.asksaveasfilename(
        defaultextension=".xlsx",
        title="Save formatted menu as",
        filetypes=[("Excel File", "*.xlsx")]
    )
    if not save_path:
        logging.warning("No save path selected. Exiting.")
        return

    if method:
        logging.info("🔍 Method: Full OCR using Tesseract")
        extracted = extract_text(input_path)
    else:
        logging.info("⚡ Method: Direct to Gemini without OCR")
        extracted = extract_raw_text(input_path)

    if not extracted:
        logging.error("Text extraction failed or returned empty.")
        return

    gemini_response = ask_gemini(extracted)
    parse_json_response(gemini_response, save_path)

    messagebox.showinfo("Done", f"Formatted menu saved to:\n{save_path}")


if __name__ == "__main__":
    main()