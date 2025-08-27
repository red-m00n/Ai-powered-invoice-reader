#!/usr/bin/env python3
"""
Test script for the fine-tuned LayoutLMv3 model
This script loads the fine-tuned model and tests it on invoice images.
"""

import os
import sys
import torch
from pathlib import Path
from PIL import Image
import numpy as np
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification
from paddleocr import PaddleOCR

# Add the app directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def load_finetuned_model(model_path: str):
    """Load the fine-tuned model and processor"""
    print(f"Loading fine-tuned model from {model_path}")
    
    try:
        # Load processor
        processor = LayoutLMv3Processor.from_pretrained(model_path)
        print("✅ Processor loaded successfully")
        
        # Load model
        model = LayoutLMv3ForTokenClassification.from_pretrained(model_path)
        print("✅ Model loaded successfully")
        
        return processor, model
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None, None

def extract_text_and_boxes(image_path: str):
    """Extract text and bounding boxes using PaddleOCR"""
    try:
        # Initialize PaddleOCR
        ocr = PaddleOCR(lang='fr', use_angle_cls=True)
        
        # Load image
        image = Image.open(image_path).convert("RGB")
        
        # Run OCR
        result = ocr.ocr(np.array(image))
        
        words = []
        boxes = []
        
        if result and result[0]:
            for line in result[0]:
                if line and len(line) >= 2:
                    try:
                        # PaddleOCR result format: [[[x1,y1,x2,y2], (text, confidence)]]
                        bbox = line[0]  # Bounding box coordinates
                        text_info = line[1]  # Text and confidence
                        
                        if len(bbox) >= 4 and len(text_info) >= 2:
                            text = text_info[0]
                            confidence = text_info[1]
                            
                            # Only keep high-confidence results
                            if confidence > 0.5:
                                words.append(text)
                                
                                # Convert coordinates to LayoutLMv3 format
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
        
        return words, boxes, image
        
    except Exception as e:
        print(f"Error in OCR processing: {e}")
        return [], [], None

def predict_fields(processor, model, image, words, boxes):
    """Predict field labels using the fine-tuned model"""
    try:
        # Prepare input
        encoding = processor(
            image,
            words,
            boxes=boxes,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )
        
        # Move to device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        for key, value in encoding.items():
            encoding[key] = value.to(device)
        
        # Get predictions
        model.eval()
        with torch.no_grad():
            outputs = model(**encoding)
            predictions_ids = torch.argmax(outputs.logits, dim=-1)
        
        # Convert to labels
        label_mapping = {
            0: "O",
            1: "INVOICE_NUMBER", 
            2: "INVOICE_DATE",
            3: "SUPPLIER_NAME",
            4: "TOTAL_HT",
            5: "TVA",
            6: "TOTAL_TTC"
        }
        
        predictions = []
        for i, pred_id in enumerate(predictions_ids[0]):
            if i < len(words):
                label = label_mapping.get(pred_id.item(), "O")
                predictions.append({
                    "word": words[i],
                    "label": label,
                    "confidence": torch.softmax(outputs.logits[0][i], dim=0).max().item()
                })
        
        return predictions
        
    except Exception as e:
        print(f"Error in prediction: {e}")
        return []

def main():
    """Main test function"""
    print("🧪 Testing Fine-tuned LayoutLMv3 Model")
    print("=" * 50)
    
    # Model path
    model_path = "./models/layoutlmv3-test"
    
    if not os.path.exists(model_path):
        print(f"❌ Model path {model_path} does not exist!")
        print("Please run the fine-tuning script first.")
        return
    
    # Load model
    processor, model = load_finetuned_model(model_path)
    if processor is None or model is None:
        return
    
    # Test on a sample invoice
    test_image_path = "../dataset/train/invoice/0.jpg"
    
    if not os.path.exists(test_image_path):
        print(f"❌ Test image {test_image_path} not found!")
        return
    
    print(f"\n📄 Testing on image: {test_image_path}")
    
    # Extract text and boxes
    words, boxes, image = extract_text_and_boxes(test_image_path)
    
    if not words:
        print("❌ No text extracted from image")
        return
    
    print(f"✅ Extracted {len(words)} words")
    print(f"Sample words: {words[:5]}")
    
    # Predict fields
    predictions = predict_fields(processor, model, image, words, boxes)
    
    if not predictions:
        print("❌ No predictions generated")
        return
    
    # Display results
    print(f"\n🎯 Field Extraction Results:")
    print("-" * 50)
    
    # Group by field type
    field_groups = {}
    for pred in predictions:
        label = pred["label"]
        if label not in field_groups:
            field_groups[label] = []
        field_groups[label].append(pred)
    
    # Display results
    for label, preds in field_groups.items():
        if label != "O":  # Skip "O" (no field) labels
            print(f"\n{label}:")
            for pred in preds:
                confidence = pred["confidence"]
                print(f"  • {pred['word']} (confidence: {confidence:.3f})")
    
    print(f"\n✅ Test completed successfully!")

if __name__ == "__main__":
    main()
