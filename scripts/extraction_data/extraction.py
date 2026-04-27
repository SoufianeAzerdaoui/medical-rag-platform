import fitz
import pdfplumber
import pytesseract
from pdf2image import convert_from_path

def extract_pdf(pdf_path):
    result = {
        "pages": [],
        "images": [],
        "tables": [],
        "raw_text": []
    }

    # Texte + tableaux via pdfplumber
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            tables = page.extract_tables()
            result["raw_text"].append({"page": i+1, "text": text})
            result["tables"].append({"page": i+1, "tables": tables})

    # Images via PyMuPDF
    doc = fitz.open(pdf_path)
    for page_num, page in enumerate(doc):
        for img in page.get_images(full=True):
            xref = img[0]
            pix = fitz.Pixmap(doc, xref)
            result["images"].append({
                "page": page_num + 1,
                "xref": xref,
                "width": pix.width,
                "height": pix.height,
                "pixmap": pix
            })

    # OCR si texte vide (page scannée)
    pages_img = convert_from_path(pdf_path, dpi=300)
    for i, img in enumerate(pages_img):
        if not result["raw_text"][i]["text"]:
            ocr_text = pytesseract.image_to_string(img, lang="fra+eng")
            result["raw_text"][i]["text"] = ocr_text
            result["raw_text"][i]["ocr_used"] = True

    return result