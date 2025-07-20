#!/usr/bin/env python3
"""
Gemini Fine-Tuning Data Formatter

Converts scraped survival data into the exact format required by Google's Gemini fine-tuning API.
Handles both text and image data with proper formatting for immediate upload.
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

class GeminiFormatter:
    """Formats survival data for Gemini fine-tuning."""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.structured_dir = self.data_dir / "structured"
        self.images_dir = self.data_dir / "images"
        self.output_dir = self.data_dir / "gemini_formatted"
        
        # Create output directory
        self.output_dir.mkdir(exist_ok=True)
        
        # Gemini system prompt for survival assistant
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
    
    def format_text_for_gemini(self, training_data: List[Dict]) -> List[Dict]:
        """Format text data for Gemini fine-tuning (JSONL format)."""
        formatted_examples = []
        
        for example in training_data:
            # Create Gemini training example format
            gemini_example = {
                "contents": [
                    {
                        "role": "user",
                        "parts": [
                            {
                                "text": example["input_text"]
                            }
                        ]
                    },
                    {
                        "role": "model", 
                        "parts": [
                            {
                                "text": example["output_text"]
                            }
                        ]
                    }
                ]
            }
            
            # Add system instruction if this is a safety-critical response
            if any(keyword in example["input_text"].lower() for keyword in ["emergency", "danger", "poison", "bite", "attack", "hypothermia", "injury"]):
                gemini_example["system_instruction"] = {
                    "parts": [
                        {
                            "text": self.system_prompt + " This is a safety-critical response - prioritize immediate life-saving actions."
                        }
                    ]
                }
            else:
                gemini_example["system_instruction"] = {
                    "parts": [
                        {
                            "text": self.system_prompt
                        }
                    ]
                }
            
            formatted_examples.append(gemini_example)
        
        return formatted_examples
    
    def encode_image_to_base64(self, image_path: Path) -> Dict[str, str]:
        """Encode image to base64 for Gemini."""
        try:
            with open(image_path, 'rb') as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
            
            # Get MIME type
            mime_type, _ = mimetypes.guess_type(str(image_path))
            if not mime_type:
                mime_type = "image/jpeg"  # Default fallback
            
            return {
                "mime_type": mime_type,
                "data": encoded_string
            }
        except Exception as e:
            logger.error(f"Error encoding image {image_path}: {e}")
            return None
    
    def create_multimodal_examples(self, image_metadata: List[Dict]) -> List[Dict]:
        """Create multimodal training examples combining images with survival context."""
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
        
        for img_data in image_metadata:
            image_path = Path(img_data["local_path"])
            
            if not image_path.exists():
                logger.warning(f"Image not found: {image_path}")
                continue
            
            # Encode image
            encoded_image = self.encode_image_to_base64(image_path)
            if not encoded_image:
                continue
            
            # Get appropriate context based on image query/title
            query = img_data.get("query", "survival")
            title = img_data.get("title", "")
            
            # Find matching context template
            contexts = context_templates.get(query, context_templates["survival camping"])
            
            # Create multiple examples per image
            for i, context_question in enumerate(contexts[:2]):  # Limit to 2 per image
                
                # Generate survival-focused response
                survival_response = self.generate_image_survival_response(
                    query, title, context_question, img_data
                )
                
                multimodal_example = {
                    "contents": [
                        {
                            "role": "user",
                            "parts": [
                                {
                                    "text": context_question
                                },
                                {
                                    "inline_data": encoded_image
                                }
                            ]
                        },
                        {
                            "role": "model",
                            "parts": [
                                {
                                    "text": survival_response
                                }
                            ]
                        }
                    ],
                    "system_instruction": {
                        "parts": [
                            {
                                "text": self.system_prompt + " Analyze the image and provide specific, actionable survival advice based on what you observe."
                            }
                        ]
                    }
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
    
    def save_gemini_training_files(self, text_examples: List[Dict], multimodal_examples: List[Dict]) -> Dict[str, str]:
        """Save formatted training files for Gemini."""
        
        output_files = {}
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. Text-only training file (JSONL format)
        text_file = self.output_dir / f"gemini_text_training_{timestamp}.jsonl"
        with open(text_file, 'w', encoding='utf-8') as f:
            for example in text_examples:
                f.write(json.dumps(example, ensure_ascii=False) + '\n')
        
        output_files['text_training'] = str(text_file)
        logger.info(f"Saved {len(text_examples)} text examples to {text_file}")
        
        # 2. Multimodal training file (JSONL format)  
        if multimodal_examples:
            multimodal_file = self.output_dir / f"gemini_multimodal_training_{timestamp}.jsonl"
            with open(multimodal_file, 'w', encoding='utf-8') as f:
                for example in multimodal_examples:
                    f.write(json.dumps(example, ensure_ascii=False) + '\n')
            
            output_files['multimodal_training'] = str(multimodal_file)
            logger.info(f"Saved {len(multimodal_examples)} multimodal examples to {multimodal_file}")
        
        # 3. Combined training file
        combined_examples = text_examples + multimodal_examples
        combined_file = self.output_dir / f"gemini_combined_training_{timestamp}.jsonl"
        with open(combined_file, 'w', encoding='utf-8') as f:
            for example in combined_examples:
                f.write(json.dumps(example, ensure_ascii=False) + '\n')
        
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
            "categories": list(set([ex.get("category", "survival") for ex in text_examples if "category" in ex])),
            "ready_for_gemini": True
        }
        
        manifest_file = self.output_dir / f"training_manifest_{timestamp}.json"
        with open(manifest_file, 'w', encoding='utf-8') as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
        
        output_files['manifest'] = str(manifest_file)
        
        return output_files
    
    def create_gemini_upload_instructions(self, output_files: Dict[str, str]) -> str:
        """Create instructions for uploading to Gemini."""
        
        instructions = f"""
# Gemini Fine-Tuning Upload Instructions

## 📁 Generated Files

Your survival training data has been formatted for Gemini fine-tuning:

- **Text Training**: `{Path(output_files['text_training']).name}`
- **Multimodal Training**: `{Path(output_files.get('multimodal_training', 'N/A')).name}`
- **Combined Training**: `{Path(output_files['combined_training']).name}`
- **Manifest**: `{Path(output_files['manifest']).name}`

## 🚀 Upload to Gemini AI Studio

### Method 1: Gemini AI Studio Web Interface

1. Go to https://aistudio.google.com/
2. Navigate to "Tuned Models" → "Create Tuned Model"
3. Upload the combined training file: `{Path(output_files['combined_training']).name}`
4. Configure settings:
   - **Model**: Gemini 1.5 Pro or Gemini 1.5 Flash
   - **Task Type**: Text Generation
   - **System Instruction**: Copy from manifest file
5. Start fine-tuning

### Method 2: Google AI SDK (Python)

```python
import google.generativeai as genai

# Configure API key
genai.configure(api_key="YOUR_API_KEY")

# Upload training data
training_data = genai.upload_file("{Path(output_files['combined_training']).name}")

# Create tuned model
model = genai.create_tuned_model(
    source_model="models/gemini-1.5-pro",
    training_data=training_data,
    id="survival-assistant",
    display_name="Wilderness Survival Assistant",
    description="Expert survival and emergency preparedness assistant"
)
```

### Method 3: gcloud CLI

```bash
# Authenticate
gcloud auth login

# Upload training data to Cloud Storage
gsutil cp {Path(output_files['combined_training']).name} gs://your-bucket/

# Start tuning job
gcloud ai custom-jobs create \\
  --region=us-central1 \\
  --display-name="survival-assistant-tuning" \\
  --config=tuning_config.yaml
```

## ⚙️ Recommended Settings

- **Learning Rate**: 0.0001 (conservative for safety-critical content)
- **Epochs**: 3-5 (avoid overfitting with small dataset)
- **Batch Size**: 4-8
- **Validation Split**: 20%

## 🧪 Testing Your Tuned Model

Test with these survival scenarios:

```
"I'm lost in the forest with no supplies. What should I do first?"
"How do I start a fire when everything is wet?"
"A bear is approaching my campsite. What do I do?"
"Someone in my group has hypothermia. How do I help?"
```

## 📊 Expected Performance

Your model should excel at:
- Wilderness survival advice
- Emergency preparedness guidance  
- Step-by-step survival instructions
- Safety-critical decision making
- Multi-modal survival scenario analysis

---
**Ready for Upload**: All files are properly formatted for Gemini fine-tuning! 🎯
"""
        
        instructions_file = self.output_dir / "UPLOAD_INSTRUCTIONS.md"
        with open(instructions_file, 'w', encoding='utf-8') as f:
            f.write(instructions)
        
        return str(instructions_file)
    
    def run_formatting(self) -> Dict[str, Any]:
        """Run complete formatting pipeline."""
        
        logger.info("🚀 Starting Gemini formatting pipeline...")
        
        # Load all data
        data = self.load_survival_data()
        
        if not data['training_data']:
            logger.error("No training data found! Run the scraper first.")
            return {}
        
        # Format text examples
        logger.info("📝 Formatting text examples for Gemini...")
        text_examples = self.format_text_for_gemini(data['training_data'])
        
        # Create multimodal examples
        logger.info("🖼️ Creating multimodal examples...")
        multimodal_examples = []
        if data['image_metadata']:
            multimodal_examples = self.create_multimodal_examples(data['image_metadata'])
        
        # Save formatted files
        logger.info("💾 Saving Gemini training files...")
        output_files = self.save_gemini_training_files(text_examples, multimodal_examples)
        
        # Create upload instructions
        instructions_file = self.create_gemini_upload_instructions(output_files)
        output_files['instructions'] = instructions_file
        
        # Summary
        total_examples = len(text_examples) + len(multimodal_examples)
        logger.info(f"✅ Formatting complete!")
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
    """Main function to run Gemini formatting."""
    
    formatter = GeminiFormatter()
    
    print("🎯 Gemini Fine-Tuning Data Formatter")
    print("=" * 50)
    
    try:
        results = formatter.run_formatting()
        
        if results:
            print(f"\n🎉 SUCCESS! Your data is now ready for Gemini fine-tuning!")
            print(f"📊 Summary:")
            print(f"  • Total training examples: {results['total_examples']}")
            print(f"  • Text-only examples: {results['text_examples']}")
            print(f"  • Multimodal examples: {results['multimodal_examples']}")
            print(f"  • Output directory: {results['output_directory']}")
            print(f"\n📖 Next steps:")
            print(f"  1. Check the generated files in {results['output_directory']}")
            print(f"  2. Read UPLOAD_INSTRUCTIONS.md for detailed upload steps")
            print(f"  3. Upload to Gemini AI Studio or use the API")
            print(f"\n🚀 Ready for immediate upload to Gemini!")
        else:
            print("❌ Formatting failed. Check the logs above for details.")
            
    except Exception as e:
        logger.error(f"Error during formatting: {e}")
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 