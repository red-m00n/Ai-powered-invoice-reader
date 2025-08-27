#!/usr/bin/env python3
"""
Test script for LayoutLMv3 fine-tuning
This script tests the fine-tuning process with a small subset of data.
"""

import os
import sys
from pathlib import Path

# Add the app directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from finetune_layoutlmv3 import InvoiceDatasetProcessor, train_layoutlmv3
from datasets import Dataset

def test_dataset_creation():
    """Test dataset creation with a small number of samples"""
    print("=== Testing Dataset Creation ===")
    
    # Initialize processor
    processor = InvoiceDatasetProcessor("../dataset/train/invoice")
    
    # Create small dataset for testing
    print("Creating test dataset with 5 samples...")
    dataset = processor.create_dataset(max_samples=5)
    
    if len(dataset) > 0:
        print(f"✅ Dataset created successfully with {len(dataset)} samples")
        print(f"Dataset features: {dataset.features}")
        
        # Show first sample
        if len(dataset) > 0:
            first_sample = dataset[0]
            print(f"\nFirst sample:")
            print(f"  Input IDs length: {len(first_sample['input_ids'])}")
            print(f"  Labels length: {len(first_sample['labels'])}")
            print(f"  Bbox length: {len(first_sample['bbox'])}")
        
        return dataset
    else:
        print("❌ Failed to create dataset")
        return None

def test_training_small(dataset):
    """Test training with a very small dataset"""
    print("\n=== Testing Small Training Run ===")
    
    if dataset is None:
        print("No dataset available for training test")
        return
    
    # Create a very small training run
    output_dir = "./models/layoutlmv3-test"
    
    try:
        # Modify training parameters for quick testing
        from finetune_layoutlmv3 import train_layoutlmv3, NUM_EPOCHS, BATCH_SIZE
        
        # Override for testing
        original_epochs = NUM_EPOCHS
        original_batch_size = BATCH_SIZE
        
        # Set to minimal values for testing
        import finetune_layoutlmv3
        finetune_layoutlmv3.NUM_EPOCHS = 1
        finetune_layoutlmv3.BATCH_SIZE = 2
        finetune_layoutlmv3.SAVE_STEPS = 10
        finetune_layoutlmv3.EVAL_STEPS = 10
        
        print("Starting minimal training run...")
        trainer = train_layoutlmv3(dataset, output_dir)
        
        print("✅ Training test completed successfully!")
        
        # Restore original values
        finetune_layoutlmv3.NUM_EPOCHS = original_epochs
        finetune_layoutlmv3.BATCH_SIZE = original_batch_size
        
        return trainer
        
    except Exception as e:
        print(f"❌ Training test failed: {e}")
        return None

def test_model_loading():
    """Test if the fine-tuned model can be loaded"""
    print("\n=== Testing Model Loading ===")
    
    model_path = "./models/layoutlmv3-test"
    
    if not os.path.exists(model_path):
        print(f"❌ Model path {model_path} does not exist")
        return False
    
    try:
        from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification
        
        print("Loading fine-tuned model...")
        processor = LayoutLMv3Processor.from_pretrained(model_path)
        model = LayoutLMv3ForTokenClassification.from_pretrained(model_path)
        
        print("✅ Model loaded successfully!")
        print(f"  Model type: {type(model)}")
        print(f"  Processor type: {type(processor)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return False

def main():
    """Main test function"""
    print("🧪 LayoutLMv3 Fine-tuning Test Suite")
    print("=" * 50)
    
    # Test 1: Dataset creation
    dataset = test_dataset_creation()
    
    if dataset is None:
        print("\n❌ Dataset creation failed. Cannot proceed with tests.")
        return
    
    # Test 2: Small training run
    trainer = test_training_small(dataset)
    
    if trainer is None:
        print("\n❌ Training test failed. Cannot proceed with model loading test.")
        return
    
    # Test 3: Model loading
    model_loaded = test_model_loading()
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY")
    print("=" * 50)
    print(f"Dataset Creation: {'✅ PASS' if dataset else '❌ FAIL'}")
    print(f"Training Test: {'✅ PASS' if trainer else '❌ FAIL'}")
    print(f"Model Loading: {'✅ PASS' if model_loaded else '❌ FAIL'}")
    
    if all([dataset, trainer, model_loaded]):
        print("\n🎉 All tests passed! Fine-tuning pipeline is working correctly.")
        print("You can now run the full fine-tuning with:")
        print("  python finetune_layoutlmv3.py")
    else:
        print("\n⚠️  Some tests failed. Check the error messages above.")

if __name__ == "__main__":
    main()
