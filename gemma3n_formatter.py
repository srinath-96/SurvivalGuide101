#!/usr/bin/env python3
"""
Gemma3n Fine-Tuning Data Formatter

Converts scraped survival data into the exact format required by Gemma3n fine-tuning.
Handles text and image data with proper formatting for Gemma3n training.
"""

import json
import os
import sys
import logging
from pathlib import Path
from typing import List, Dict, Any
import base64
import mimetypes
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Gemma3nFormatter:
    """Formats survival data for Gemma3n fine-tuning."""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.structured_dir = self.data_dir / "structured"
        self.images_dir = self.data_dir / "images"
        self.output_dir = self.data_dir / "gemma3n_formatted"
        
        # Create output directory
        self.output_dir.mkdir(exist_ok=True)
        
        # Create images directory for Gemma3n
        (self.output_dir / "images").mkdir(exist_ok=True)
        
        # Gemma3n system prompt for survival assistant
        self.system_prompt = """You are an expert wilderness survival instructor and emergency preparedness specialist. Your role is to provide accurate, life-saving advice for outdoor emergencies and survival situations. Always prioritize safety, provide step-by-step instructions, and include important warnings about potential dangers. Base your responses on proven survival techniques and emergency preparedness best practices."""
    
    def load_survival_data(self) -> Dict[str, Any]:
        """Load all survival data files."""
        data = {
            'text_data': [],
            'training_data': [],
            'image_metadata': []
        }
        
        # Load comprehensive survival data
        comprehensive_file = self.structured_dir / "comprehensive_survival_data.json"
        if comprehensive_file.exists():
            with open(comprehensive_file, 'r', encoding='utf-8') as f:
                data['text_data'] = json.load(f)
            logger.info(f"Loaded {len(data['text_data'])} text items")
        
        # Load training dataset
        training_file = self.structured_dir / "gemini_training_dataset.json"
        if training_file.exists():
            with open(training_file, 'r', encoding='utf-8') as f:
                data['training_data'] = json.load(f)
            logger.info(f"Loaded {len(data['training_data'])} training examples")
        
        # Load image metadata
        image_metadata_file = self.structured_dir / "fallback_image_metadata.json"
        if image_metadata_file.exists():
            with open(image_metadata_file, 'r', encoding='utf-8') as f:
                data['image_metadata'] = json.load(f)
            logger.info(f"Loaded {len(data['image_metadata'])} image metadata entries")
        
        return data
    
    def decode_and_save_image(self, base64_data: str, image_filename: str) -> str:
        """Decode base64 image and save to file for Gemma3n."""
        try:
            # Remove data:image header if present
            if base64_data.startswith('data:'):
                base64_data = base64_data.split(',')[1]
            
            # Decode and save
            image_data = base64.b64decode(base64_data)
            image_path = self.output_dir / "images" / image_filename
            
            with open(image_path, 'wb') as f:
                f.write(image_data)
            
            return str(image_path.relative_to(self.output_dir))
        except Exception as e:
            logger.error(f"Error decoding image {image_filename}: {e}")
            return None
    
    def format_text_for_gemma3n(self, training_data: List[Dict]) -> List[Dict]:
        """Format text data for Gemma3n fine-tuning."""
        formatted_examples = []
        
        for idx, example in enumerate(training_data):
            # Create Gemma3n training example format
            gemma3n_example = {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text", 
                                "text": example["input_text"]
                            }
                        ]
                    },
                    {
                        "role": "assistant", 
                        "content": [
                            {
                                "type": "text", 
                                "text": example["output_text"]
                            }
                        ]
                    }
                ],
                "system_prompt": self.system_prompt,
                "example_id": f"text_{idx}"
            }
            
            formatted_examples.append(gemma3n_example)
        
        return formatted_examples
    
    def create_multimodal_examples(self, image_metadata: List[Dict]) -> List[Dict]:
        """Create multimodal training examples for Gemma3n."""
        multimodal_examples = []
        
        # Survival context templates for different image types
        context_templates = {
            "survival camping": [
                "What survival techniques are shown in this camping scenario?",
                "How would you set up a survival camp based on what you see in this image?",
                "What survival priorities should be considered in this camping situation?"
            ],
            "fire making": [
                "Describe the fire-making technique shown in this image.",
                "What are the key elements for successful fire creation that you can identify here?",
                "How would you improve or modify this fire setup for better survival efficiency?"
            ],
            "water purification": [
                "What water purification methods could be applied in this scenario?",
                "How would you assess the water safety in this environment?",
                "What steps would you take to make this water source safe for drinking?"
            ],
            "wilderness first aid": [
                "What first aid considerations are relevant to this wilderness situation?",
                "How would you handle a medical emergency in this environment?",
                "What medical supplies would be most important in this setting?"
            ],
            "emergency shelter": [
                "How would you build an emergency shelter in this environment?",
                "What natural materials could you use for shelter construction here?",
                "What are the key shelter priorities for this type of terrain?"
            ],
            "survival tools": [
                "What improvised tools could you create in this environment?",
                "How would you prioritize tool creation for survival in this setting?",
                "What natural materials could serve as survival tools here?"
            ]
        }
        
        for idx, img_data in enumerate(image_metadata):
            image_path = Path(img_data["local_path"])
            
            if not image_path.exists():
                logger.warning(f"Image not found: {image_path}")
                continue
            
            # Copy image to Gemma3n format directory
            image_filename = f"survival_image_{idx}.jpg"
            destination = self.output_dir / "images" / image_filename
            
            try:
                import shutil
                shutil.copy2(image_path, destination)
                relative_image_path = f"images/{image_filename}"
            except Exception as e:
                logger.error(f"Error copying image {image_path}: {e}")
                continue
            
            # Get appropriate context based on image query/title
            query = img_data.get("query", "survival")
            title = img_data.get("title", "")
            
            # Find matching context template
            contexts = context_templates.get(query, context_templates["survival camping"])
            
            # Create examples for this image
            for i, context_question in enumerate(contexts[:2]):  # Limit to 2 per image
                
                # Generate survival-focused response
                survival_response = self.generate_image_survival_response(
                    query, title, context_question, img_data
                )
                
                multimodal_example = {
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": context_question
                                },
                                {
                                    "type": "image",
                                    "url": relative_image_path
                                }
                            ]
                        },
                        {
                            "role": "assistant",
                            "content": [
                                {
                                    "type": "text",
                                    "text": survival_response
                                }
                            ]
                        }
                    ],
                    "system_prompt": self.system_prompt,
                    "example_id": f"multimodal_{idx}_{i}"
                }
                
                multimodal_examples.append(multimodal_example)
        
        return multimodal_examples
    
    def generate_image_survival_response(self, query: str, title: str, question: str, img_data: Dict) -> str:
        """Generate contextual survival response for image."""
        
        # Base responses for different survival categories
        response_templates = {
            "survival camping": f"""Based on this camping scenario, here are key survival considerations:

1. **Shelter Assessment**: Look for natural windbreaks and elevated, dry ground. Avoid low areas where cold air settles.

2. **Fire Safety**: Maintain safe distance from flammable materials. Create a fire ring with stones if possible.

3. **Water Access**: Identify nearest water sources but always purify before drinking - boil for at least 1 minute.

4. **Food Storage**: Keep food secured and away from sleeping areas to avoid attracting wildlife.

5. **Emergency Signaling**: Maintain visibility for potential rescue - use bright colors and clear sightlines.

6. **Weather Protection**: Prepare for temperature drops and precipitation changes typical in wilderness environments.

Remember: The survival rule of 3 - 3 minutes without air, 3 hours without shelter in harsh conditions, 3 days without water, 3 weeks without food.""",

            "fire making": f"""This fire-making scenario demonstrates several important survival principles:

1. **Fire Triangle**: Ensure adequate heat source, fuel (tinder, kindling, fuel wood), and oxygen flow.

2. **Material Preparation**: Gather materials in three sizes:
   - Tinder: Fine, dry material (grass, bark, paper)
   - Kindling: Pencil-thick to thumb-thick dry wood
   - Fuel wood: Progressively larger pieces

3. **Fire Structure**: Build a foundation to keep fire off wet ground. Create airflow channels.

4. **Safety Measures**: Clear area of flammable debris in 10-foot radius. Have water/dirt nearby for extinguishing.

5. **Efficiency Tips**: Start small and build up gradually. Blow gently at base to increase oxygen flow.

6. **Weather Adaptations**: In wet conditions, look for dead branches still on trees, split wood to access dry interior.

Fire provides warmth, water purification, cooking, signaling, and psychological comfort in survival situations.""",

            "water purification": f"""Water safety is critical for survival. Here's how to approach this water source:

1. **Source Assessment**: Moving water is generally safer than stagnant. Avoid water near animal trails, campsites, or industrial areas.

2. **Physical Filtration**: Remove large particles using cloth, sand, charcoal, and gravel layers.

3. **Purification Methods**:
   - Boiling: Most reliable - rolling boil for 1+ minutes
   - Chemical: Iodine or chlorine tablets per package instructions
   - UV: Clear water with UV sterilization if available
   - Solar: Clear bottles in direct sunlight for 6+ hours

4. **Collection**: Let sediment settle before treatment. Collect from surface or spring sources when possible.

5. **Storage**: Use clean containers. Mark treated vs untreated water clearly.

6. **Rationing**: Adults need minimum 1 liter/day, more in hot climates or high activity.

Never assume any natural water source is safe to drink untreated, even if it appears clean."""
        }
        
        # Get appropriate template or use camping as default
        template_key = query if query in response_templates else "survival camping"
        return response_templates[template_key]
    
    def save_gemma3n_training_files(self, text_examples: List[Dict], multimodal_examples: List[Dict]) -> Dict[str, str]:
        """Save formatted training files for Gemma3n."""
        
        output_files = {}
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. Text-only training file (JSON format for Gemma3n)
        text_file = self.output_dir / f"gemma3n_text_training_{timestamp}.json"
        with open(text_file, 'w', encoding='utf-8') as f:
            json.dump(text_examples, f, indent=2, ensure_ascii=False)
        
        output_files['text_training'] = str(text_file)
        logger.info(f"Saved {len(text_examples)} text examples to {text_file}")
        
        # 2. Multimodal training file (JSON format for Gemma3n)  
        if multimodal_examples:
            multimodal_file = self.output_dir / f"gemma3n_multimodal_training_{timestamp}.json"
            with open(multimodal_file, 'w', encoding='utf-8') as f:
                json.dump(multimodal_examples, f, indent=2, ensure_ascii=False)
            
            output_files['multimodal_training'] = str(multimodal_file)
            logger.info(f"Saved {len(multimodal_examples)} multimodal examples to {multimodal_file}")
        
        # 3. Combined training file
        combined_examples = text_examples + multimodal_examples
        combined_file = self.output_dir / f"gemma3n_combined_training_{timestamp}.json"
        with open(combined_file, 'w', encoding='utf-8') as f:
            json.dump(combined_examples, f, indent=2, ensure_ascii=False)
        
        output_files['combined_training'] = str(combined_file)
        logger.info(f"Saved {len(combined_examples)} total examples to {combined_file}")
        
        # 4. Create training manifest
        manifest = {
            "created_at": datetime.now().isoformat(),
            "total_examples": len(combined_examples),
            "text_examples": len(text_examples),
            "multimodal_examples": len(multimodal_examples),
            "files": output_files,
            "system_prompt": self.system_prompt,
            "format": "gemma3n",
            "ready_for_training": True
        }
        
        manifest_file = self.output_dir / f"gemma3n_manifest_{timestamp}.json"
        with open(manifest_file, 'w', encoding='utf-8') as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
        
        output_files['manifest'] = str(manifest_file)
        
        return output_files
    
    def create_gemma3n_training_script(self, output_files: Dict[str, str]) -> str:
        """Create a training script for Gemma3n."""
        
        training_script = f'''#!/usr/bin/env python3
"""
Gemma3n Survival Assistant Fine-Tuning Script
Generated automatically from survival data formatter.
"""

import json
import os
import torch
from datasets import Dataset
from transformers import AutoProcessor, Gemma3nForConditionalGeneration
from trl import SFTConfig, SFTTrainer
from peft import LoraConfig
from PIL import Image

def load_survival_dataset():
    """Load the formatted survival training data."""
    with open("{Path(output_files['combined_training']).name}", 'r') as f:
        training_data = json.load(f)
    
    return Dataset.from_list(training_data)

def collate_fn(examples):
    """Collate function for Gemma3n survival training."""
    example = examples[0]
    messages = example["messages"]
    
    # Apply chat template
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=False,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        padding=True,
    ).to(model.device)
    
    # Create labels for training
    labels = inputs["input_ids"].clone()
    special_token_ids = processor.tokenizer.all_special_ids
    special_token_ids_tensor = torch.tensor(special_token_ids, device=labels.device)
    mask = torch.isin(labels, special_token_ids_tensor)
    labels[mask] = -100
    
    inputs["labels"] = labels
    return inputs

# Load model and processor
model = Gemma3nForConditionalGeneration.from_pretrained(
    "google/gemma-3n-E2B-it", 
    torch_dtype=torch.bfloat16,
)
processor = AutoProcessor.from_pretrained("google/gemma-3n-E2B-it")
processor.tokenizer.padding_side = "right"

# Load dataset
dataset = load_survival_dataset()
dataset = dataset.train_test_split(test_size=0.1, seed=42)

# LoRA configuration
peft_config = LoraConfig(
    task_type="CAUSAL_LM",
    r=16,
    target_modules="all-linear",
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    use_rslora=False,
    use_dora=False,
)

# Training configuration
training_args = SFTConfig(
    output_dir="./gemma3n-survival-assistant",
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
    dataset_kwargs={{'skip_prepare_dataset': True}},
    remove_unused_columns=False,
    max_seq_length=None,
    dataloader_pin_memory=False,
)

# Setup trainer
model.config.use_cache = False
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
print("🚀 Starting Gemma3n Survival Assistant fine-tuning...")
trainer.train()

print("✅ Training completed! Model saved to ./gemma3n-survival-assistant")
'''
        
        script_file = self.output_dir / "train_gemma3n_survival.py"
        with open(script_file, 'w', encoding='utf-8') as f:
            f.write(training_script)
        
        return str(script_file)
    
    def run_formatting(self) -> Dict[str, Any]:
        """Run complete Gemma3n formatting pipeline."""
        
        logger.info("🚀 Starting Gemma3n formatting pipeline...")
        
        # Load all data
        data = self.load_survival_data()
        
        if not data['training_data']:
            logger.error("No training data found! Run the scraper first.")
            return {}
        
        # Format text examples
        logger.info("📝 Formatting text examples for Gemma3n...")
        text_examples = self.format_text_for_gemma3n(data['training_data'])
        
        # Create multimodal examples
        logger.info("🖼️ Creating multimodal examples...")
        multimodal_examples = []
        if data['image_metadata']:
            multimodal_examples = self.create_multimodal_examples(data['image_metadata'])
        
        # Save formatted files
        logger.info("💾 Saving Gemma3n training files...")
        output_files = self.save_gemma3n_training_files(text_examples, multimodal_examples)
        
        # Create training script
        training_script = self.create_gemma3n_training_script(output_files)
        output_files['training_script'] = training_script
        
        # Summary
        total_examples = len(text_examples) + len(multimodal_examples)
        logger.info(f"✅ Gemma3n formatting complete!")
        logger.info(f"📊 Total examples: {total_examples}")
        logger.info(f"📝 Text examples: {len(text_examples)}")
        logger.info(f"🖼️ Multimodal examples: {len(multimodal_examples)}")
        logger.info(f"📁 Files saved to: {self.output_dir}")
        
        return {
            'total_examples': total_examples,
            'text_examples': len(text_examples),
            'multimodal_examples': len(multimodal_examples),
            'output_files': output_files,
            'output_directory': str(self.output_dir)
        }


def main():
    """Main function to run Gemma3n formatting."""
    
    formatter = Gemma3nFormatter()
    
    print("🎯 Gemma3n Survival Assistant Data Formatter")
    print("=" * 50)
    
    try:
        results = formatter.run_formatting()
        
        if results:
            print(f"\n🎉 SUCCESS! Your data is now ready for Gemma3n fine-tuning!")
            print(f"📊 Summary:")
            print(f"  • Total training examples: {results['total_examples']}")
            print(f"  • Text-only examples: {results['text_examples']}")
            print(f"  • Multimodal examples: {results['multimodal_examples']}")
            print(f"  • Output directory: {results['output_directory']}")
            print(f"\n📖 Next steps:")
            print(f"  1. Run: python {Path(results['output_files']['training_script']).name}")
            print(f"  2. Monitor training progress with tensorboard")
            print(f"  3. Test your survival assistant model!")
            print(f"\n🚀 Ready for Gemma3n training!")
        else:
            print("❌ Formatting failed. Check the logs above for details.")
            
    except Exception as e:
        logger.error(f"Error during formatting: {e}")
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 