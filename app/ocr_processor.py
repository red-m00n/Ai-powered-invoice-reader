#!/usr/bin/env python3
"""
Clean OCR Processor with Fine-tuned LayoutLMv3
This script extracts OCR text and uses the fine-tuned model to extract invoice fields.
"""

import sys
import os
import re
from datetime import datetime, date
from pdf2image import convert_from_path
from PIL import Image
from app.db import SessionLocal, engine
from app.models import Invoice, Base
from paddleocr import PaddleOCR
import numpy as np
import torch
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification

# Ensure database tables exist
Base.metadata.create_all(bind=engine)

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize PaddleOCR
ocr = PaddleOCR(lang='fr', use_textline_orientation=True)

# Model paths - resolve relative to this file to avoid CWD issues
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "layoutlmv3-test")

# Field labels for the fine-tuned model
FIELD_LABELS = ["O", "INVOICE_NUMBER", "INVOICE_DATE", "SUPPLIER_NAME", "TOTAL_HT", "TVA", "TOTAL_TTC"]

def load_finetuned_model():
    """Load the fine-tuned LayoutLMv3 model and processor"""
    try:
        if not os.path.exists(MODEL_PATH):
            print(f"❌ Fine-tuned model not found at {MODEL_PATH}")
            print("Please run the fine-tuning script first.")
            return None, None
        
        print(f"Loading fine-tuned model from {MODEL_PATH}")
        
        # Load processor first
        processor = LayoutLMv3Processor.from_pretrained(MODEL_PATH)
        print("✅ Processor loaded successfully")
        
        # Load model with correct configuration
        model = LayoutLMv3ForTokenClassification.from_pretrained(MODEL_PATH)
        print("✅ Model loaded successfully")
        
        # Move to device
        model.to(device)
        model.eval()
        
        return processor, model
        
    except Exception as e:
        print(f"❌ Error loading fine-tuned model: {e}")
        return None, None

def extract_text_and_boxes(image):
    """Extract text and bounding boxes from image using PaddleOCR"""
    try:
        img_np = np.array(image)
        print(f"Image size: {image.size}")
        
        # Use predict instead of ocr for newer PaddleOCR versions
        try:
            result = ocr.predict(img_np)
        except AttributeError:
            result = ocr.ocr(img_np)
        
        print(f"OCR result type: {type(result)}")
        
        words = []
        boxes = []
        
        if result and len(result) > 0:
            # Handle different PaddleOCR result formats
            if isinstance(result[0], dict):
                # New format with metadata
                ocr_result = result[0]
                if 'rec_texts' in ocr_result:
                    texts = ocr_result['rec_texts']
                    bboxes = ocr_result.get('rec_boxes', [])
                    scores = ocr_result.get('rec_scores', [])
                    
                    print(f"Found {len(texts)} text items")
                    
                    for i, text in enumerate(texts):
                        if i < len(bboxes):
                            bbox = bboxes[i]
                            confidence = scores[i] if i < len(scores) else 1.0
                            
                            if confidence > 0.3:  # Lower threshold for testing
                                words.append(text)
                                
                                # Convert coordinates to LayoutLMv3 format
                                if len(bbox) >= 4:
                                    x0, y0, x1, y1 = bbox
                                    width, height = image.size
                                    normalized_box = [
                                        max(0, min(1000, int((x0 * 1000) // width))),
                                        max(0, min(1000, int((y0 * 1000) // height))),
                                        max(0, min(1000, int((x1 * 1000) // width))),
                                        max(0, min(1000, int((y1 * 1000) // height)))
                                    ]
                                    boxes.append(normalized_box)
            else:
                # Traditional format
                for line in result[0]:
                    if line and len(line) >= 2:
                        try:
                            bbox = line[0]
                            text_info = line[1]
                            
                            if len(bbox) >= 4 and len(text_info) >= 2:
                                text = text_info[0]
                                confidence = text_info[1]
                                
                                if confidence > 0.3:
                                    words.append(text)
                                    
                                    x0, y0, x1, y1 = bbox
                                    width, height = image.size
                                    normalized_box = [
                                        max(0, min(1000, int((x0 * 1000) // width))),
                                        max(0, min(1000, int((y0 * 1000) // height))),
                                        max(0, min(1000, int((x1 * 1000) // width))),
                                        max(0, min(1000, int((y1 * 1000) // height)))
                                    ]
                                    boxes.append(normalized_box)
                        except Exception as e:
                            print(f"Error processing OCR line: {e}")
                            continue
        
        print(f"Final words: {words}")
        return words, boxes
        
    except Exception as e:
        print(f"Error in OCR processing: {e}")
        import traceback
        traceback.print_exc()
        return [], []

def predict_fields_with_layoutlmv3(processor, model, image, words, boxes):
    """Use the fine-tuned LayoutLMv3 model to predict field labels"""
    try:
        if not words or not boxes:
            return {}
        
        # Prepare input for the model
        encoding = processor(
            images=image, 
            text=words, 
            boxes=boxes, 
            return_tensors="pt"
        )
        
        # Move to device
        encoding = {k: v.to(device) for k, v in encoding.items()}
        
        # Get predictions
        with torch.no_grad():
            outputs = model(**encoding)
            predictions = torch.argmax(outputs.logits, dim=-1).squeeze().tolist()
        
        # Extract fields based on predictions
        extracted_fields = {
            "invoice_number": [],
            "invoice_date": [],
            "supplier_name": [],
            "total_ht": [],
            "tva": [],
            "total_ttc": []
        }
        
        for word, pred_id in zip(words, predictions):
            if pred_id < len(FIELD_LABELS):
                label = FIELD_LABELS[pred_id]
                if label != "O":  # Skip "Other" label
                    extracted_fields[label.lower()].append({"text": word, "confidence": 0.9})
        
        return extracted_fields
        
    except Exception as e:
        print(f"Error in LayoutLMv3 prediction: {e}")
        import traceback
        traceback.print_exc()
        return {}

def extract_fields_fallback(words):
    """Fallback field extraction using regex patterns when model fails"""
    extracted_fields = {
        "invoice_number": [],
        "invoice_date": [],
        "supplier_name": [],
        "total_ht": [],
        "tva": [],
        "total_ttc": []
    }
    
    text_lower = [word.lower() for word in words]
    
    for i, word in enumerate(text_lower):
        # Invoice number patterns
        if any(pattern in word for pattern in ['pièce', 'n°', 'facture', 'invoice', 'fact', 'no', 'num']):
            if any(char.isdigit() for char in word):
                extracted_fields["invoice_number"].append({"text": words[i], "confidence": 0.8})
        
        # Date patterns - look for actual dates, not phone numbers
        elif any(pattern in word for pattern in ['/']):
            if any(char.isdigit() for char in word) and len(word) >= 8:
                if word.count('/') >= 2:
                    extracted_fields["invoice_date"].append({"text": words[i], "confidence": 0.8})
        
        # Amount patterns
        elif any(pattern in word for pattern in ['total', 'ht', 'ttc', 'tva', '€', 'euros']):
            if any(char.isdigit() for char in word):
                if 'ht' in word or 'h.t' in word:
                    extracted_fields["total_ht"].append({"text": words[i], "confidence": 0.8})
                elif 'tva' in word:
                    extracted_fields["tva"].append({"text": words[i], "confidence": 0.8})
                elif 'ttc' in word or 'total' in word:
                    extracted_fields["total_ttc"].append({"text": words[i], "confidence": 0.8})
        
        # Supplier name
        elif i < 5 and not any(char.isdigit() for char in word) and len(word) > 3:
            if word not in ['éléphone', 'téléphone', 'fax', 'code', 'page', 'désignation', 'prix', 'unité', 'qté', 'remise', 'montant', 'taux', 'base', 'dont', 'autre', 'total', 'payer', 'joindre', 'règlement', 'membre']:
                extracted_fields["supplier_name"].append({"text": words[i], "confidence": 0.7})
    
    # Post-process to find better matches
    for i, word in enumerate(words):
        word_lower = word.lower()
        
        # Look for actual invoice number
        if word.isdigit() and len(word) >= 4:
            extracted_fields["invoice_number"].append({"text": word, "confidence": 0.9})
        
        # Look for actual amounts
        elif any(char.isdigit() for char in word) and any(char in word for char in [',', '.', '€']):
            if ':' in word or '/' in word or len(word) > 20:
                continue
                
            if any(char in word for char in [',', '.', '€']) and len(word) <= 15:
                try:
                    clean_amount = word.replace('€', '').replace(',', '.').strip()
                    if clean_amount.replace('.', '').replace('-', '').isdigit():
                        amount_value = float(clean_amount)
                        if amount_value > 100:
                            extracted_fields["total_ttc"].append({"text": word, "confidence": 0.9})
                        else:
                            extracted_fields["total_ht"].append({"text": word, "confidence": 0.8})
                except:
                    extracted_fields["total_ttc"].append({"text": word, "confidence": 0.7})
    
    return extracted_fields

def combine_fields(fields_list):
    """Combine fields from multiple images"""
    if not fields_list:
        return {}
    
    combined = {
        "invoice_number": [],
        "invoice_date": [],
        "supplier_name": [],
        "total_ht": [],
        "tva": [],
        "total_ttc": []
    }
    
    for fields in fields_list:
        for key in combined:
            if key in fields and fields[key]:
                combined[key].extend(fields[key])
    
    # Remove duplicates and keep highest confidence
    for key in combined:
        if combined[key]:
            seen = set()
            unique_fields = []
            for field in sorted(combined[key], key=lambda x: x["confidence"], reverse=True):
                if field["text"] not in seen:
                    seen.add(field["text"])
                    unique_fields.append(field)
            combined[key] = unique_fields
    
    return combined

def extract_best_field(field_list):
    """Extract the best field value from a list of field candidates"""
    if not field_list:
        return None
    
    # Return the field with highest confidence
    best_field = max(field_list, key=lambda x: x["confidence"])
    return best_field["text"]

def parse_amount(amount_str):
    """Parse amount strings to float values for database storage"""
    if not amount_str:
        return None
    
    try:
        # Remove currency symbols and spaces
        cleaned = str(amount_str).replace('€', '').replace(' ', '').strip()
        # Replace comma with dot for decimal (French format)
        cleaned = cleaned.replace(',', '.')
        # Extract numeric value
        amount_match = re.search(r'[\d\-]+\.?\d*', cleaned)
        if amount_match:
            return float(amount_match.group())
        return None
    except Exception:
        return None

def parse_invoice_date(date_str):
    """Parse various French/ISO date strings to a Python date object."""
    if not date_str:
        return None
    try:
        text = str(date_str)
        # Remove labels like 'DATE :', 'ÉCHÉANCE :', etc.
        text = re.sub(r"(?i)(date|échéance|due|le|du)\s*[:\-]?\s*", "", text).strip()
        # Common formats: DD / MM / YYYY, DD/MM/YYYY, YYYY-MM-DD
        # Try yyyy-mm-dd
        for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%d / %m / %Y", "%d-%m-%Y", "%d.%m.%Y"):
            try:
                return datetime.strptime(text, fmt).date()
            except Exception:
                pass
        # Fallback: extract digits
        m = re.search(r"(\d{1,2})\s*[\./\-\s]?\s*(\d{1,2})\s*[\./\-\s]?\s*(\d{2,4})", text)
        if m:
            d, mth, y = m.groups()
            if len(y) == 2:
                y = "20" + y
            return date(int(y), int(mth), int(d))
        return None
    except Exception:
        return None

def save_to_database(extracted_fields, file_path, ocr_text):
    """Save extracted fields to database"""
    try:
        db = SessionLocal()
        
        # Check if invoice already exists
        existing_invoice = db.query(Invoice).filter(Invoice.filename == os.path.basename(file_path)).first()
        if existing_invoice:
            print(f"Invoice {os.path.basename(file_path)} already exists in database")
            print(f"Existing invoice ID: {existing_invoice.id}")
            db.close()
            return existing_invoice
        
        # Create new invoice record
        parsed_date = parse_invoice_date(extract_best_field(extracted_fields["invoice_date"]))
        invoice = Invoice(
            filename=os.path.basename(file_path),
            invoice_number=extract_best_field(extracted_fields["invoice_number"]),
            invoice_date=parsed_date,
            supplier_name=extract_best_field(extracted_fields["supplier_name"]),
            total_ht=parse_amount(extract_best_field(extracted_fields["total_ht"])),
            tva=parse_amount(extract_best_field(extracted_fields["tva"])),
            total_ttc=parse_amount(extract_best_field(extracted_fields["total_ttc"])),
            ocr_text=ocr_text
        )
        
        db.add(invoice)
        db.commit()
        db.refresh(invoice)
        db.close()
        
        print(f"✅ Invoice saved to database: {os.path.basename(file_path)}")
        print(f"   ID: {invoice.id}")
        return invoice
        
    except Exception as e:
        print(f"Error saving to database: {e}")
        import traceback
        traceback.print_exc()
        if 'db' in locals():
            db.rollback()
            db.close()
        return None

def process_invoice_file(file_path):
    """Main function to process invoice file"""
    try:
        print(f"Processing file: {file_path}")
        
        # Load the fine-tuned model
        processor, model = load_finetuned_model()
        
        # Convert PDF to images if needed
        if file_path.lower().endswith('.pdf'):
            images = convert_from_path(file_path)
            print(f"Converted PDF to {len(images)} images")
        else:
            # Load image directly
            images = [Image.open(file_path).convert('RGB')]
        
        all_extracted_fields = []
        ocr_text_parts = []
        
        for i, img in enumerate(images):
            print(f"Processing image {i+1}/{len(images)}")
            
            # Extract text and bounding boxes
            words, boxes = extract_text_and_boxes(img)
            
            if not words:
                print(f"No text extracted from image {i+1}")
                continue
            
            print(f"Extracted {len(words)} words from image {i+1}")
            
            # Try to use LayoutLMv3 model first
            if processor and model:
                print("Using fine-tuned LayoutLMv3 model")
                extracted_fields = predict_fields_with_layoutlmv3(processor, model, img, words, boxes)
                
                # If model didn't extract enough fields, use fallback
                if not any(extracted_fields.values()):
                    print("Model extraction failed, using regex fallback")
                    extracted_fields = extract_fields_fallback(words)
            else:
                print("Using regex extraction (model not available)")
                extracted_fields = extract_fields_fallback(words)
            
            print(f"Extracted fields: {extracted_fields}")
            
            all_extracted_fields.append(extracted_fields)
            ocr_text_parts.append(' '.join(words))
        
        # Combine results from all images
        combined_fields = combine_fields(all_extracted_fields)
        full_ocr_text = '\n'.join(ocr_text_parts)
        
        print("Combined extracted fields:")
        for key, value in combined_fields.items():
            print(f"  {key}: {value}")
        
        # Save to database
        result = save_to_database(combined_fields, file_path, full_ocr_text)
        
        return result
        
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    if len(sys.argv) != 2:
        print("Usage: python ocr_processor.py uploads/nom_de_la_facture.pdf")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    if not os.path.exists(pdf_path):
        print(f"File not found: {pdf_path}")
        sys.exit(1)
    
    # Ensure database tables exist
    Base.metadata.create_all(bind=engine)
    
    result = process_invoice_file(pdf_path)
    if result:
        print(f"Successfully processed: {pdf_path}")
    else:
        print(f"Failed to process: {pdf_path}")

if __name__ == "__main__":
    main()
