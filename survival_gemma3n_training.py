#!/usr/bin/env python3
"""
Survival Assistant Gemma3n Fine-Tuning Script
Adapted from the FineVideo notebook for survival data training.
"""

import json
import os
import torch
from datasets import Dataset
from transformers import AutoProcessor, Gemma3nForConditionalGeneration
from trl import SFTConfig, SFTTrainer
from peft import LoraConfig
from PIL import Image
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_survival_dataset(data_file: str):
    """Load the formatted survival training data."""
    with open(data_file, 'r', encoding='utf-8') as f:
        training_data = json.load(f)
    
    logger.info(f"Loaded {len(training_data)} training examples")
    return Dataset.from_list(training_data)

def collate_fn(examples):
    """
    Collate function for Gemma3n survival training.
    Handles both text-only and multimodal examples.
    """
    example = examples[0]
    messages = example["messages"]
    
    # Apply chat template - this handles the image loading automatically
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=False,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        padding=True,
    ).to(model.device)
    
    # Create labels for training (mask special tokens)
    labels = inputs["input_ids"].clone()
    special_token_ids = processor.tokenizer.all_special_ids
    special_token_ids_tensor = torch.tensor(special_token_ids, device=labels.device)
    mask = torch.isin(labels, special_token_ids_tensor)
    labels[mask] = -100
    
    inputs["labels"] = labels
    
    # Debug info
    if torch.all(inputs.get("pixel_values", torch.tensor([1])) == 0):
        logger.warning("No image data found in this batch")
    
    return inputs

def setup_model_and_processor(model_name: str = "google/gemma-3n-E2B-it"):
    """Setup Gemma3n model and processor."""
    logger.info(f"Loading model: {model_name}")
    
    model = Gemma3nForConditionalGeneration.from_pretrained(
        model_name, 
        torch_dtype=torch.bfloat16,
    )
    
    processor = AutoProcessor.from_pretrained(model_name)
    processor.tokenizer.padding_side = "right"
    
    # Disable cache for training
    model.config.use_cache = False
    
    return model, processor

def create_lora_config():
    """Create LoRA configuration for efficient fine-tuning."""
    return LoraConfig(
        task_type="CAUSAL_LM",
        r=16,
        target_modules="all-linear",
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        use_rslora=False,
        use_dora=False,
    )

def create_training_config(output_dir: str = "./gemma3n-survival-assistant"):
    """Create training configuration."""
    return SFTConfig(
        output_dir=output_dir,
        eval_strategy='epoch',
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,
        gradient_checkpointing=False,
        learning_rate=1e-05,
        num_train_epochs=3.0,
        logging_steps=10,
        save_steps=100,
        bf16=True,
        report_to=["tensorboard"],
        dataset_kwargs={'skip_prepare_dataset': True},
        remove_unused_columns=False,
        max_seq_length=None,
        dataloader_pin_memory=False,
        save_total_limit=2,
        load_best_model_at_end=True,
    )

def main():
    """Main training function."""
    
    print("🎯 Survival Assistant Gemma3n Fine-Tuning")
    print("=" * 50)
    
    # Configuration
    DATA_FILE = "data/gemma3n_formatted/gemma3n_combined_training_*.json"
    OUTPUT_DIR = "./gemma3n-survival-assistant"
    
    # Find the latest training file
    import glob
    data_files = glob.glob(DATA_FILE)
    if not data_files:
        print("❌ No training data found! Run gemma3n_formatter.py first.")
        return
    
    latest_data_file = max(data_files)
    print(f"📂 Using training data: {latest_data_file}")
    
    # Load model and processor
    global model, processor
    model, processor = setup_model_and_processor()
    
    # Load dataset
    print("📊 Loading survival dataset...")
    dataset = load_survival_dataset(latest_data_file)
    
    # Split dataset
    dataset = dataset.train_test_split(test_size=0.1, seed=42)
    print(f"📈 Training examples: {len(dataset['train'])}")
    print(f"🧪 Test examples: {len(dataset['test'])}")
    
    # Sample data inspection
    sample = dataset['train'][0]
    print(f"\n🔍 Sample training example:")
    print(f"  Messages: {len(sample['messages'])}")
    print(f"  User content types: {[c['type'] for c in sample['messages'][0]['content']]}")
    
    # Setup training components
    peft_config = create_lora_config()
    training_args = create_training_config(OUTPUT_DIR)
    
    # Create trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        data_collator=collate_fn,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        processing_class=processor.tokenizer,
        peft_config=peft_config,
    )
    
    # Start training
    print("\n🚀 Starting fine-tuning...")
    print("💡 Monitor progress with: tensorboard --logdir ./gemma3n-survival-assistant/logs")
    
    try:
        trainer.train()
        print("✅ Training completed successfully!")
        
        # Save final model
        trainer.save_model()
        print(f"💾 Model saved to: {OUTPUT_DIR}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        print(f"❌ Training failed: {e}")
        return
    
    # Test the model
    print("\n🧪 Testing the trained model...")
    test_model(trainer.model, processor)

def test_model(model, processor):
    """Test the trained survival assistant."""
    
    test_questions = [
        "I'm lost in the forest with no supplies. What should I do first?",
        "How do I start a fire when everything is wet?",
        "What are the signs of hypothermia and how do I treat it?",
        "How can I find safe drinking water in the wilderness?",
        "What should I do if I encounter a bear?"
    ]
    
    print("🎯 Testing Survival Assistant:")
    print("-" * 40)
    
    for question in test_questions:
        # Format as chat
        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": question}]
            }
        ]
        
        # Process input
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)
        
        # Generate response
        with torch.inference_mode():
            input_len = inputs["input_ids"].shape[-1]
            generation = model.generate(
                **inputs, 
                max_new_tokens=150, 
                do_sample=False,
                temperature=0.1
            )
            generation = generation[0][input_len:]
        
        response = processor.decode(generation, skip_special_tokens=True)
        
        print(f"\nQ: {question}")
        print(f"A: {response}")
        print("-" * 40)

if __name__ == "__main__":
    main() 