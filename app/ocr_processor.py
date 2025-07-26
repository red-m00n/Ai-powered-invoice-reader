import sys
import os
import re
from datetime import datetime
from pdf2image import convert_from_path
from PIL import Image
from app.db import SessionLocal, engine
from app.models import Invoice, Base
from paddleocr import PaddleOCR
import numpy as np

def pdf_to_images(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == '.pdf':
        return convert_from_path(file_path)
    else:
        return [Image.open(file_path)]

def ocr_images(images):
    ocr = PaddleOCR(lang='fr', use_angle_cls=True)
    full_text = []
    for img in images:
        img_np = np.array(img)
        result = ocr.ocr(img_np, cls=True)
        for line in result:
            for word_info in line:
                txt = word_info[1][0]
                full_text.append(txt)
    return '\n'.join(full_text)

def extract_invoice_fields(ocr_text):
    invoice_number = None
    invoice_date = None
    supplier_name = None
    total_ht = None
    tva = None
    total_ttc = None


    matches = re.findall(r'^(?:Facture|Pi[èe]ce)\s*[Nn][o°]?(?:\s*[:\-])?\s*([A-Za-z0-9\-/ ]+)$', ocr_text, re.IGNORECASE | re.MULTILINE)
    if matches:
 
        invoice_number = max(matches, key=lambda x: len(x.strip())).strip()
    else:

        matches = re.findall(r'^(?:Facture|Pi[èe]ce)\s*[:\-]?\s*([A-Za-z0-9\-/ ]+)$', ocr_text, re.IGNORECASE | re.MULTILINE)
        if matches:
            invoice_number = max(matches, key=lambda x: len(x.strip())).strip()


    match = re.search(r'(\d{2})\s*/\s*(\d{2})\s*/\s*(\d{4})', ocr_text)
    if match:
        try:
            date_str = f"{match.group(1)}/{match.group(2)}/{match.group(3)}"
            invoice_date = datetime.strptime(date_str, '%d/%m/%Y').date()
        except Exception:
            invoice_date = None


    match = re.search(r'Total\s*H[.\s]?T[.\s]?(?:\.|\s)?\s*[:\-]?\s*([\d\s,.]+)', ocr_text, re.IGNORECASE)
    if not match:
        match = re.search(r'H[.\s]?T[.\s]?(?:\.|\s)?\s*[:\-]?\s*([\d\s,.]+)', ocr_text, re.IGNORECASE)
    if match:
        total_ht = parse_amount(match.group(1))


    match = re.search(r'T\.?\s*V\.?\s*A\.?[^\d\n]*[:\-]?\s*([\d\s,.]+)', ocr_text, re.IGNORECASE)
    if not match:
        match = re.search(r'TVA[^\d\n]*[:\-]?\s*([\d\s,.]+)', ocr_text, re.IGNORECASE)
    if match:
        tva = parse_amount(match.group(1))


    match = re.search(r'Total\s*T[.\s]?T[.\s]?C[.\s]?(?:\.|\s)?\s*[:\-]?\s*([\d\s,.]+)', ocr_text, re.IGNORECASE)
    if not match:
        match = re.search(r'T[.\s]?T[.\s]?C[.\s]?(?:\.|\s)?\s*[:\-]?\s*([\d\s,.]+)', ocr_text, re.IGNORECASE)
    if match:
        total_ttc = parse_amount(match.group(1))


    lines = [l.strip() for l in ocr_text.splitlines() if l.strip()]
    if lines:
        supplier_name = lines[0][:255]

    return {
        'invoice_number': invoice_number,
        'invoice_date': invoice_date,
        'supplier_name': supplier_name,
        'total_ht': total_ht,
        'tva': tva,
        'total_ttc': total_ttc
    }

def parse_amount(amount_str):
    try:
        cleaned = amount_str.replace(' ', '').replace(',', '.').replace('€', '')
        return float(re.findall(r'[\d.]+', cleaned)[0])
    except Exception:
        return None

def save_to_db(filename, ocr_text, fields):
    session = SessionLocal()
    try:
        invoice = Invoice(
            filename=filename,
            invoice_number=fields['invoice_number'],
            invoice_date=fields['invoice_date'],
            supplier_name=fields['supplier_name'],
            total_ht=fields['total_ht'],
            tva=fields['tva'],
            total_ttc=fields['total_ttc'],
            ocr_text=ocr_text
        )
        session.add(invoice)
        session.commit()
        print(f"Résultat inséré dans la base pour {filename}")
    except Exception as e:
        session.rollback()
        print("Erreur lors de l'insertion en base :", e)
    finally:
        session.close()

    
def process_invoice_file(file_path):


    print(f"Conversion du PDF en images...")
    images = pdf_to_images(file_path)
    print(f"Extraction OCR en cours...")
    ocr_text = ocr_images(images)
    print("Texte extrait :\n", ocr_text)

    fields = extract_invoice_fields(ocr_text)
    print("Champs extraits :")
    for k, v in fields.items():
        print(f"{k}: {v}")


    return save_to_db(os.path.basename(file_path), ocr_text, fields)


def main():
    if len(sys.argv) != 2:
        print("Usage: python main.py uploads/nom_de_la_facture.pdf")
        sys.exit(1)
    pdf_path = sys.argv[1]
    if not os.path.exists(pdf_path):
        print(f"Fichier non trouvé: {pdf_path}")
        sys.exit(1)

    # Crée les tables si elles n'existent pas
    Base.metadata.create_all(bind=engine)

    print(f"Conversion du PDF en images...")
    images = pdf_to_images(pdf_path)
    print(f"Extraction OCR en cours...")
    ocr_text = ocr_images(images)
    print("Texte extrait :\n")
    print(ocr_text)
    fields = extract_invoice_fields(ocr_text)
    print("Champs extraits :")
    for k, v in fields.items():
        print(f"{k}: {v}")
    save_to_db(os.path.basename(pdf_path), ocr_text, fields)

if __name__ == "__main__":
    main() 