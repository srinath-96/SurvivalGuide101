#!/usr/bin/env python3
"""
Gemma3n Fine-Tuning Data Formatter (Fixed Version)

Converts scraped survival data into the exact format required by Gemma3n fine-tuning.
Handles text and image data with proper base64 encoding for Gemma3n training.
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
from PIL import Image
import io

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Gemma3nFormatter:
    """Formats survival data for Gemma3n fine-tuning with proper image handling."""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.structured_dir = self.data_dir / "structured"
        self.images_dir = self.data_dir / "images"
        self.output_dir = self.data_dir / "gemma3n_formatted"
        
        # Create output directory
        self.output_dir.mkdir(exist_ok=True)
        
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

    def encode_image_to_base64(self, image_path: str) -> str:
        """
        Encode image to base64 with data URI format for Gemma3n.
        Returns a data URI string like: data:image/jpeg;base64,/9j/4AAQSkZJRg...
        """
        try:
            # Open and convert image to RGB JPEG
            with Image.open(image_path) as img:
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Save as JPEG to bytes buffer
                buffer = io.BytesIO()
                img.save(buffer, format='JPEG')
                img_bytes = buffer.getvalue()
            
            # Encode to base64
            base64_encoded = base64.b64encode(img_bytes).decode('utf-8')
            
            # Return as data URI
            return f"data:image/jpeg;base64,{base64_encoded}"
            
        except Exception as e:
            logger.error(f"Error encoding image {image_path}: {e}")
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
        """Create multimodal training examples with base64-encoded images."""
        multimodal_examples = []
        
        # Context templates for different types of survival scenarios
        context_templates = {
            "survival camping": [
                "What survival skills would be most important in this environment?",
                "What natural resources in this image could be useful for survival?",
                "What potential dangers should one be aware of in this setting?"
            ],
            "wilderness navigation": [
                "What natural navigation markers can you identify in this image?",
                "How would you use the terrain features shown for orientation?",
                "What direction-finding techniques would work best here?"
            ],
            "shelter building": [
                "How would you construct a shelter using materials visible in this environment?",
                "What natural features could provide emergency shelter here?",
                "What shelter-building considerations are important in this terrain?"
            ],
            "water sources": [
                "What potential water sources can you identify in this image?",
                "How would you make any water found here safe to drink?",
                "What water collection methods would work best in this environment?"
            ]
        }
        
        for idx, img_data in enumerate(image_metadata):
            image_path = Path(img_data["local_path"])
            
            if not image_path.exists():
                logger.warning(f"Image not found: {image_path}")
                continue
            
            # Encode image to base64 data URI
            image_data_uri = self.encode_image_to_base64(str(image_path))
            if not image_data_uri:
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
                                    "source": {
                                        "type": "base64",
                                        "data": image_data_uri
                                    }
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
        # Template responses for different query types
        response_templates = {
            "survival camping": (
                "In this environment, several key survival considerations stand out. "
                "First, notice the available natural resources that could be useful: {resources}. "
                "For shelter, you could {shelter_method}. "
                "The terrain suggests that {water_source} might be available for water. "
                "Important safety considerations include {safety_points}. "
                "Remember to {key_reminder} in this type of environment."
            ).format(
                resources="fallen branches, leaves, and natural formations",
                shelter_method="use the natural formations for wind protection and construct a lean-to shelter",
                water_source="seasonal streams or collected rainwater",
                safety_points="weather exposure, wildlife activity, and terrain hazards",
                key_reminder="maintain awareness of your surroundings and preserve energy"
            ),
            
            "wilderness navigation": (
                "For navigation in this terrain, there are several key features to note. "
                "The {landmarks} can serve as reference points. "
                "You can determine direction by {direction_method}. "
                "Important navigational considerations include {nav_points}. "
                "Remember to {nav_reminder} when traversing this area."
            ).format(
                landmarks="natural formations and distinctive features",
                direction_method="observing the sun's position and natural indicators",
                nav_points="maintaining line of sight with landmarks and watching for terrain changes",
                nav_reminder="regularly confirm your heading and maintain awareness of your starting position"
            )
        }
        
        # Get appropriate template or use camping as default
        template_key = query if query in response_templates else "survival camping"
        return response_templates[template_key]

    def save_gemma3n_training_files(self, text_examples: List[Dict], multimodal_examples: List[Dict]) -> Dict[str, str]:
        """Save formatted training files for Gemma3n."""
        
        output_files = {}
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. Text-only training file
        text_file = self.output_dir / f"gemma3n_text_training_{timestamp}.json"
        with open(text_file, 'w', encoding='utf-8') as f:
            json.dump(text_examples, f, indent=2, ensure_ascii=False)
        
        output_files['text_training'] = str(text_file)
        logger.info(f"Saved {len(text_examples)} text examples to {text_file}")
        
        # 2. Multimodal training file
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
        
        # 4. Create manifest file
        manifest = {
            'timestamp': timestamp,
            'total_examples': len(combined_examples),
            'text_examples': len(text_examples),
            'multimodal_examples': len(multimodal_examples),
            'files': output_files
        }
        
        manifest_file = self.output_dir / f"gemma3n_manifest_{timestamp}.json"
        with open(manifest_file, 'w', encoding='utf-8') as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
        
        output_files['manifest'] = str(manifest_file)
        
        return output_files

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

if __name__ == "__main__":
    formatter = Gemma3nFormatter()
    results = formatter.run_formatting()
    print(json.dumps(results, indent=2)) 