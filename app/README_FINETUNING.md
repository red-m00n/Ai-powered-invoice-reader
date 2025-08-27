# LayoutLMv3 Fine-tuning for Invoice Field Extraction

This directory contains the complete pipeline for fine-tuning LayoutLMv3 on your invoice dataset to extract key fields like invoice numbers, dates, amounts, and supplier names.

## 🎯 Overview

The system consists of three main components:

1. **`finetune_layoutlmv3.py`** - Main fine-tuning script
2. **`ocr_processor_finetuned.py`** - OCR processor using the fine-tuned model
3. **`test_finetuning.py`** - Test script to verify the pipeline

## 🚀 Quick Start

### 1. Install Dependencies

First, install the additional dependencies needed for fine-tuning:

```bash
pip install -r requirements.txt
```

### 2. Test the Pipeline

Run the test script to verify everything works:

```bash
cd app
python test_finetuning.py
```

This will:
- Create a small test dataset (5 samples)
- Run a minimal training session (1 epoch)
- Verify the model can be loaded

### 3. Run Full Fine-tuning

Once the test passes, run the full fine-tuning:

```bash
python finetune_layoutlmv3.py
```

This will:
- Process your entire invoice dataset
- Fine-tune LayoutLMv3 for 10 epochs
- Save the model to `./models/layoutlmv3-invoice/`
- Evaluate the model performance

### 4. Use the Fine-tuned Model

Update your main.py to use the fine-tuned processor:

```python
# In main.py, change the import from:
from app.ocr_processor import process_invoice_file

# To:
from app.ocr_processor_finetuned import process_invoice_file
```

## 📁 File Structure

```
app/
├── finetune_layoutlmv3.py          # Main fine-tuning script
├── ocr_processor_finetuned.py      # OCR processor with fine-tuned model
├── test_finetuning.py              # Test script
├── models/                         # Output directory for trained models
│   └── layoutlmv3-invoice/        # Fine-tuned model files
└── README_FINETUNING.md            # This file
```

## ⚙️ Configuration

### Training Parameters

Edit `finetune_layoutlmv3.py` to adjust training parameters:

```python
# Training configuration
BATCH_SIZE = 4              # Reduce if you run out of GPU memory
LEARNING_RATE = 5e-5        # Learning rate for fine-tuning
NUM_EPOCHS = 10             # Number of training epochs
WARMUP_STEPS = 500          # Warmup steps for learning rate
SAVE_STEPS = 1000           # Save model every N steps
EVAL_STEPS = 500            # Evaluate every N steps
```

### Field Labels

The model is trained to recognize these invoice fields:

```python
FIELD_LABELS = [
    "O",              # Outside any field
    "INVOICE_NUMBER", # Invoice number/ID
    "INVOICE_DATE",   # Invoice date
    "SUPPLIER_NAME",  # Company/supplier name
    "TOTAL_HT",      # Total before tax
    "TVA",           # Tax amount
    "TOTAL_TTC"      # Total including tax
]
```

## 🔧 How It Works

### 1. Data Preparation

The script automatically:
- Loads your invoice images from `../dataset/train/invoice/`
- Uses PaddleOCR to extract text and bounding boxes
- Creates synthetic labels based on text patterns
- Prepares data in LayoutLMv3 format

### 2. Model Training

- Loads pre-trained LayoutLMv3 base model
- Fine-tunes on your invoice data
- Saves checkpoints during training
- Evaluates performance on validation set

### 3. Field Extraction

The fine-tuned model:
- Processes new invoice images
- Predicts field labels for each text token
- Extracts structured data (numbers, dates, amounts)
- Falls back to regex patterns if needed

## 📊 Expected Results

After fine-tuning, you should see:

- **Better field recognition** than the base model
- **Improved accuracy** for invoice-specific patterns
- **Faster processing** for similar invoice types
- **More reliable extraction** of amounts and dates

## 🐛 Troubleshooting

### Common Issues

1. **Out of Memory (OOM)**
   - Reduce `BATCH_SIZE` in the configuration
   - Use fewer training samples initially

2. **Training Too Slow**
   - Reduce `NUM_EPOCHS` for testing
   - Use a subset of your dataset first

3. **Model Not Loading**
   - Check if the model directory exists
   - Verify all files were saved correctly

4. **Poor Performance**
   - Increase training data quality
   - Adjust learning rate
   - Train for more epochs

### Debug Mode

Enable debug output by modifying the logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 🔄 Integration with Your App

### Option 1: Replace OCR Processor

```python
# In main.py
from app.ocr_processor_finetuned import process_invoice_file

# The rest of your code remains the same
```

### Option 2: Hybrid Approach

```python
# Use fine-tuned model when available, fallback to regex
try:
    from app.ocr_processor_finetuned import process_invoice_file
except ImportError:
    from app.ocr_processor import process_invoice_file
```

## 📈 Performance Monitoring

The training script provides:

- **Training loss** over time
- **Validation metrics** (precision, recall, F1)
- **Model checkpoints** for best performance
- **Evaluation results** saved to JSON

## 🎉 Next Steps

1. **Run the test script** to verify setup
2. **Fine-tune on your data** for better accuracy
3. **Integrate the model** into your OCR pipeline
4. **Monitor performance** and retrain if needed
5. **Expand the dataset** with more invoice types

## 📚 Additional Resources

- [LayoutLMv3 Paper](https://arxiv.org/abs/2204.08387)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/)
- [PaddleOCR Documentation](https://github.com/PaddlePaddle/PaddleOCR)

## 🤝 Support

If you encounter issues:

1. Check the error messages in the console
2. Verify your dataset path and format
3. Ensure all dependencies are installed
4. Try with a smaller dataset first

The system is designed to be robust and will fall back to regex-based extraction if the ML model fails.
