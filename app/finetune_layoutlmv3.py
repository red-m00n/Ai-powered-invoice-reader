#!/usr/bin/env python3
"""
LayoutLMv3 Fine-tuning Script for Invoice Field Extraction
This script fine-tunes LayoutLMv3 on invoice images to extract key fields.
"""

import os
import json
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from typing import List, Dict, Tuple
from datetime import datetime
import random
from tqdm import tqdm

# ML/AI imports
from transformers import (
    LayoutLMv3Processor, 
    LayoutLMv3ForTokenClassification,
    TrainingArguments, 
    Trainer,
    DataCollatorForTokenClassification
)
from datasets import Dataset, Features, Value, Sequence, ClassLabel
import evaluate

# OCR imports
from paddleocr import PaddleOCR

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Configuration
DATASET_PATH = "../dataset/train/invoice"
OUTPUT_DIR = "./models/layoutlmv3-invoice"
BATCH_SIZE = 4
LEARNING_RATE = 5e-5
NUM_EPOCHS = 10
WARMUP_STEPS = 500
SAVE_STEPS = 1000
EVAL_STEPS = 500

# Field labels for invoice extraction
FIELD_LABELS = [
    "O",  # Outside any field
    "INVOICE_NUMBER",
    "INVOICE_DATE", 
    "SUPPLIER_NAME",
    "TOTAL_HT",
    "TVA",
    "TOTAL_TTC"
]

# Create label2id and id2label mappings
label2id = {label: idx for idx, label in enumerate(FIELD_LABELS)}
id2label = {idx: label for label, idx in label2id.items()}

print(f"Field labels: {FIELD_LABELS}")
print(f"Label mappings: {label2id}")

class InvoiceDatasetProcessor:
    """Process invoice images and create training data for LayoutLMv3"""
    
    def __init__(self, dataset_path: str):
        self.dataset_path = Path(dataset_path)
        self.ocr = PaddleOCR(lang='fr', use_angle_cls=True)
        # Initialize processor without OCR since we're providing our own
        self.processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)
        
    def extract_text_and_boxes(self, image_path: str) -> Tuple[List[str], List[List[int]]]:
        """Extract text and bounding boxes from image using PaddleOCR"""
        try:
            image = Image.open(image_path).convert("RGB")
            result = self.ocr.ocr(np.array(image))
            
            words = []
            boxes = []
            
            if result and result[0]:
                # PaddleOCR returns a list of dictionaries with metadata
                ocr_result = result[0]
                
                # Extract text from rec_texts field
                if 'rec_texts' in ocr_result:
                    texts = ocr_result['rec_texts']
                    print(f"Found {len(texts)} text items: {texts}")
                    
                    # Extract bounding boxes from rec_boxes field
                    if 'rec_boxes' in ocr_result:
                        bboxes = ocr_result['rec_boxes']
                        print(f"Found {len(bboxes)} bounding boxes")
                        
                        # Extract confidence scores
                        scores = ocr_result.get('rec_scores', [])
                        
                        # Process each text item
                        for i, text in enumerate(texts):
                            if i < len(bboxes):
                                bbox = bboxes[i]
                                confidence = scores[i] if i < len(scores) else 1.0
                                
                                print(f"  Text: '{text}', Confidence: {confidence}, Box: {bbox}")
                                
                                # Only keep high-confidence results
                                if confidence > 0.5:
                                    words.append(text)
                                    
                                    # Convert coordinates to LayoutLMv3 format
                                    # bbox format: [x1, y1, x2, y2]
                                    x0, y0, x1, y1 = bbox
                                    # Normalize coordinates to [0, 1000] using proper integer arithmetic
                                    width, height = image.size
                                    normalized_box = [
                                        max(0, min(1000, int((x0 * 1000) // width))),
                                        max(0, min(1000, int((y0 * 1000) // height))),
                                        max(0, min(1000, int((x1 * 1000) // width))),
                                        max(0, min(1000, int((y1 * 1000) // height)))
                                    ]
                                    boxes.append(normalized_box)
                                    print(f"  Added word: '{text}' with box {normalized_box}")
                else:
                    print(f"No rec_texts found in OCR result")
            else:
                print(f"No OCR result for {image_path}")
            
            print(f"Final words: {words}")
            print(f"Final boxes: {boxes}")
            return words, boxes
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return [], []
    
    def create_synthetic_annotations(self, words: List[str], boxes: List[List[int]]) -> List[int]:
        """Create synthetic labels for training data"""
        if not words:
            return []
        
        labels = []
        text_lower = [word.lower() for word in words]
        
        for i, word in enumerate(text_lower):
            label = "O"  # Default label
            
            # Invoice number patterns
            if any(pattern in word for pattern in ['facture', 'invoice', 'fact', 'no', 'num']):
                label = "INVOICE_NUMBER"
            # Date patterns
            elif any(pattern in word for pattern in ['date', 'le', 'du', '/', '-', '.']):
                if any(char.isdigit() for char in word):
                    label = "INVOICE_DATE"
            # Amount patterns
            elif any(pattern in word for pattern in ['total', 'ht', 'ttc', 'tva', '€', 'euros']):
                if any(char.isdigit() for char in word):
                    if 'ht' in word or 'h.t' in word:
                        label = "TOTAL_HT"
                    elif 'tva' in word:
                        label = "TVA"
                    elif 'ttc' in word or 'total' in word:
                        label = "TOTAL_TTC"
            # Supplier name (first few words without numbers)
            elif i < 3 and not any(char.isdigit() for char in word) and len(word) > 3:
                label = "SUPPLIER_NAME"
            
            labels.append(label2id[label])
        
        return labels
    
    def process_image(self, image_path: str) -> Dict:
        """Process a single image and return training data"""
        words, boxes = self.extract_text_and_boxes(image_path)
        
        if not words:
            return None
        
        # Create synthetic labels
        labels = self.create_synthetic_annotations(words, boxes)
        
        # Prepare image for processor
        image = Image.open(image_path).convert("RGB")
        
        # Process with LayoutLMv3 processor
        encoding = self.processor(
            image,
            words,
            boxes=boxes,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )
        
        return {
            "input_ids": encoding["input_ids"].squeeze().tolist(),
            "attention_mask": encoding["attention_mask"].squeeze().tolist(),
            "bbox": encoding["bbox"].squeeze().tolist(),
            "labels": labels[:len(encoding["input_ids"].squeeze())]  # Truncate labels if needed
        }
    
    def create_dataset(self, max_samples: int = None) -> Dataset:
        """Create training dataset from invoice images"""
        print("Creating training dataset...")
        
        # Get all image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(self.dataset_path.glob(f"*{ext}"))
        
        if max_samples:
            image_files = image_files[:max_samples]
        
        print(f"Found {len(image_files)} images")
        
        # Process images
        dataset_data = []
        for image_path in tqdm(image_files, desc="Processing images"):
            data = self.process_image(str(image_path))
            if data:
                dataset_data.append(data)
        
        print(f"Successfully processed {len(dataset_data)} images")
        
        # Create dataset
        features = Features({
            "input_ids": Sequence(Value("int64")),
            "attention_mask": Sequence(Value("int64")),
            "bbox": Sequence(Sequence(Value("int64"))),
            "labels": Sequence(Value("int64"))
        })
        
        return Dataset.from_list(dataset_data, features=features)

def train_layoutlmv3(dataset: Dataset, output_dir: str):
    """Fine-tune LayoutLMv3 model"""
    print("Initializing model for fine-tuning...")
    
    # Load pre-trained model
    model = LayoutLMv3ForTokenClassification.from_pretrained(
        "microsoft/layoutlmv3-base",
        num_labels=len(FIELD_LABELS),
        id2label=id2label,
        label2id=label2id
    )
    
    # Move to device
    model.to(device)
    
    # Split dataset
    dataset = dataset.train_test_split(test_size=0.2, seed=42)
    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Evaluation samples: {len(eval_dataset)}")
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        warmup_steps=WARMUP_STEPS,
        save_steps=SAVE_STEPS,
        eval_steps=EVAL_STEPS,
        save_strategy="steps",
        logging_dir=f"{output_dir}/logs",
        logging_steps=100,
        dataloader_num_workers=0,  # Set to 0 for Windows compatibility
        remove_unused_columns=False,
        push_to_hub=False,
        report_to=None,
    )
    
    # Initialize processor
    processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base")
    
    # Data collator
    data_collator = DataCollatorForTokenClassification(
        tokenizer=processor.tokenizer,
        return_tensors="pt"
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=processor,
    )
    
    # Train the model
    print("Starting training...")
    trainer.train()
    
    # Save the model
    print(f"Saving model to {output_dir}")
    trainer.save_model(output_dir)
    
    # Save processor
    processor.save_pretrained(output_dir)
    
    # Save configuration
    config = {
        "field_labels": FIELD_LABELS,
        "label2id": label2id,
        "id2label": id2label,
        "training_args": training_args.to_dict(),
        "dataset_info": {
            "train_samples": len(train_dataset),
            "eval_samples": len(eval_dataset),
            "total_samples": len(train_dataset) + len(eval_dataset)
        }
    }
    
    with open(f"{output_dir}/config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print("Training completed!")
    return trainer

def evaluate_model(model_path: str, test_dataset: Dataset):
    """Evaluate the fine-tuned model"""
    print(f"Evaluating model from {model_path}")
    
    # Load fine-tuned model
    model = LayoutLMv3ForTokenClassification.from_pretrained(model_path)
    model.to(device)
    model.eval()
    
    # Load processor
    processor = LayoutLMv3Processor.from_pretrained(model_path)
    
    # Metrics
    metric = evaluate.load("seqeval")
    
    # Evaluation
    predictions = []
    true_labels = []
    
    print("Running evaluation...")
    for sample in tqdm(test_dataset, desc="Evaluating"):
        # Prepare input
        encoding = processor(
            return_tensors="pt",
            truncation=True,
            max_length=512
        )
        
        # Move to device
        for key, value in encoding.items():
            encoding[key] = value.to(device)
        
        # Get predictions
        with torch.no_grad():
            outputs = model(**encoding)
            predictions_ids = torch.argmax(outputs.logits, dim=-1)
        
        # Convert to labels
        pred_labels = [id2label[id.item()] for id in predictions_ids[0]]
        true_label = [id2label[id] for id in sample["labels"]]
        
        predictions.append(pred_labels)
        true_labels.append(true_label)
    
    # Calculate metrics
    results = metric.compute(predictions=predictions, references=true_labels)
    
    print("Evaluation Results:")
    for key, value in results.items():
        print(f"  {key}: {value:.4f}")
    
    return results

def main():
    """Main function"""
    print("=== LayoutLMv3 Invoice Field Extraction Fine-tuning ===")
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Initialize processor
    processor = InvoiceDatasetProcessor(DATASET_PATH)
    
    # Create dataset (limit to 100 samples for testing)
    print("Creating dataset...")
    dataset = processor.create_dataset(max_samples=100)
    
    if len(dataset) == 0:
        print("No valid samples found. Check your dataset path and image files.")
        return
    
    # Train model
    print("Starting fine-tuning...")
    trainer = train_layoutlmv3(dataset, OUTPUT_DIR)
    
    # Evaluate model
    print("Evaluating fine-tuned model...")
    test_dataset = dataset.train_test_split(test_size=0.2, seed=42)["test"]
    results = evaluate_model(OUTPUT_DIR, test_dataset)
    
    # Save results
    with open(f"{OUTPUT_DIR}/evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nFine-tuning completed! Model saved to: {OUTPUT_DIR}")
    print("You can now use this model in your OCR processor.")

if __name__ == "__main__":
    main()
