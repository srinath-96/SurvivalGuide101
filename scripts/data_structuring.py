"""
Data structuring and preprocessing utilities for survival data.
"""

import os
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataStructurer:
    """Class for structuring and preprocessing scraped survival data."""
    
    def __init__(self, data_paths: Dict[str, str]):
        self.data_paths = data_paths
        self.structured_path = Path(data_paths['structured'])
        
    def structure_image_data(self) -> Dict[str, Any]:
        """Structure and validate image metadata."""
        image_metadata_file = self.structured_path / 'image_metadata.json'
        
        if not image_metadata_file.exists():
            logger.warning("No image metadata file found")
            return {}
        
        with open(image_metadata_file, 'r') as f:
            images = json.load(f)
        
        # Create structured dataset
        structured_images = []
        for img in images:
            structured_img = {
                'id': img.get('id'),
                'filename': f"{img.get('source', 'unknown')}_{img.get('id', 'unknown')}.jpg",
                'description': img.get('description', img.get('alt_description', '')),
                'tags': img.get('tags', []),
                'query': img.get('query', ''),
                'source': img.get('source', ''),
                'width': img.get('width', 0),
                'height': img.get('height', 0),
                'aspect_ratio': img.get('width', 0) / max(img.get('height', 1), 1),
                'category': self._categorize_image(img)
            }
            structured_images.append(structured_img)
        
        # Save structured data
        output_file = self.structured_path / 'structured_images.json'
        with open(output_file, 'w') as f:
            json.dump(structured_images, f, indent=2)
        
        logger.info(f"Structured {len(structured_images)} images")
        return {'images': structured_images, 'count': len(structured_images)}
    
    def structure_text_data(self) -> Dict[str, Any]:
        """Structure and validate text content."""
        text_data_file = self.structured_path / 'survival_text_data.json'
        
        if not text_data_file.exists():
            logger.warning("No text data file found")
            return {}
        
        with open(text_data_file, 'r') as f:
            texts = json.load(f)
        
        # Create structured dataset
        structured_texts = []
        for text in texts:
            structured_text = {
                'id': self._generate_text_id(text),
                'title': text.get('title', ''),
                'content': self._extract_content(text),
                'source': text.get('source', ''),
                'category': self._categorize_text(text),
                'tags': self._extract_tags(text),
                'url': text.get('url', ''),
                'scraped_at': text.get('scraped_at', '')
            }
            structured_texts.append(structured_text)
        
        # Save structured data
        output_file = self.structured_path / 'structured_texts.json'
        with open(output_file, 'w') as f:
            json.dump(structured_texts, f, indent=2)
        
        logger.info(f"Structured {len(structured_texts)} text items")
        return {'texts': structured_texts, 'count': len(structured_texts)}
    
    def _categorize_image(self, img: Dict) -> str:
        """Categorize image based on query and description."""
        query = img.get('query', '').lower()
        description = img.get('description', '').lower()
        
        categories = {
            'shelter': ['shelter', 'tent', 'cabin', 'hut'],
            'fire': ['fire', 'flame', 'campfire', 'lighter'],
            'water': ['water', 'purification', 'filter', 'stream'],
            'food': ['food', 'cooking', 'hunting', 'fishing'],
            'tools': ['tools', 'knife', 'axe', 'gear'],
            'medical': ['medical', 'first aid', 'emergency'],
            'navigation': ['navigation', 'compass', 'map', 'gps']
        }
        
        text = f"{query} {description}"
        for category, keywords in categories.items():
            if any(keyword in text for keyword in keywords):
                return category
        
        return 'general'
    
    def _categorize_text(self, text: Dict) -> str:
        """Categorize text content based on title and content."""
        title = text.get('title', '').lower()
        content = text.get('content', '').lower()
        
        categories = {
            'shelter': ['shelter', 'tent', 'cabin', 'build', 'construct'],
            'fire': ['fire', 'ignite', 'flint', 'lighter', 'matches'],
            'water': ['water', 'purify', 'filter', 'boil', 'stream'],
            'food': ['food', 'cook', 'hunt', 'fish', 'forage'],
            'tools': ['tools', 'knife', 'axe', 'make', 'diy'],
            'medical': ['medical', 'first aid', 'injury', 'wound', 'treatment'],
            'navigation': ['navigation', 'compass', 'map', 'gps', 'lost']
        }
        
        text_content = f"{title} {content}"
        for category, keywords in categories.items():
            if any(keyword in text_content for keyword in keywords):
                return category
        
        return 'general'
    
    def _extract_content(self, text: Dict) -> str:
        """Extract main content from text data."""
        content_parts = []
        
        if text.get('introduction'):
            content_parts.append(text['introduction'])
        
        if text.get('steps'):
            for step in text['steps']:
                if step.get('title'):
                    content_parts.append(step['title'])
                if step.get('content'):
                    content_parts.append(step['content'])
        
        if text.get('tips'):
            content_parts.append("Tips: " + "; ".join(text['tips']))
        
        if text.get('warnings'):
            content_parts.append("Warnings: " + "; ".join(text['warnings']))
        
        return " ".join(content_parts)
    
    def _extract_tags(self, text: Dict) -> List[str]:
        """Extract relevant tags from text content."""
        content = self._extract_content(text).lower()
        
        survival_keywords = [
            'survival', 'wilderness', 'emergency', 'preparedness', 'bushcraft',
            'camping', 'hiking', 'outdoor', 'shelter', 'fire', 'water', 'food',
            'tools', 'medical', 'navigation', 'safety', 'diy', 'tutorial'
        ]
        
        tags = [keyword for keyword in survival_keywords if keyword in content]
        return tags[:5]  # Limit to top 5 tags
    
    def _generate_text_id(self, text: Dict) -> str:
        """Generate unique ID for text content."""
        title = text.get('title', '')
        url = text.get('url', '')
        return f"{text.get('source', 'unknown')}_{hash(title + url) % 1000000}"
    
    def create_training_dataset(self) -> Dict[str, Any]:
        """Create formatted dataset for model training."""
        # Load structured data
        images_file = self.structured_path / 'structured_images.json'
        texts_file = self.structured_path / 'structured_texts.json'
        
        if not images_file.exists() or not texts_file.exists():
            logger.error("Structured data files not found")
            return {}
        
        with open(images_file, 'r') as f:
            images = json.load(f)
        
        with open(texts_file, 'r') as f:
            texts = json.load(f)
        
        # Create training dataset
        training_data = {
            'metadata': {
                'created_at': datetime.now().isoformat(),
                'total_images': len(images),
                'total_texts': len(texts),
                'categories': list(set([img['category'] for img in images] + 
                                     [text['category'] for text in texts]))
            },
            'images': images,
            'texts': texts
        }
        
        # Save training dataset
        output_file = self.structured_path / 'survival_training_dataset.json'
        with open(output_file, 'w') as f:
            json.dump(training_data, f, indent=2)
        
        logger.info("Created training dataset")
        return training_data
    
    def generate_summary_report(self) -> Dict[str, Any]:
        """Generate summary report of collected data."""
        report = {
            'timestamp': datetime.now().isoformat(),
            'images': {},
            'texts': {},
            'categories': {}
        }
        
        # Analyze images
        images_file = self.structured_path / 'structured_images.json'
        if images_file.exists():
            with open(images_file, 'r') as f:
                images = json.load(f)
            
            report['images'] = {
                'total': len(images),
                'by_source': {},
                'by_category': {},
                'avg_aspect_ratio': sum([img['aspect_ratio'] for img in images]) / len(images) if images else 0
            }
            
            for img in images:
                source = img['source']
                category = img['category']
                report['images']['by_source'][source] = report['images']['by_source'].get(source, 0) + 1
                report['images']['by_category'][category] = report['images']['by_category'].get(category, 0) + 1
        
        # Analyze texts
        texts_file = self.structured_path / 'structured_texts.json'
        if texts_file.exists():
            with open(texts_file, 'r') as f:
                texts = json.load(f)
            
            report['texts'] = {
                'total': len(texts),
                'by_source': {},
                'by_category': {},
                'avg_length': sum([len(text['content']) for text in texts]) / len(texts) if texts else 0
            }
            
            for text in texts:
                source = text['source']
                category = text['category']
                report['texts']['by_source'][source] = report['texts']['by_source'].get(source, 0) + 1
                report['texts']['by_category'][category] = report['texts']['by_category'].get(category, 0) + 1
        
        # Save report
        report_file = self.structured_path / 'data_summary_report.json'
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        return report


if __name__ == "__main__":
    import sys
    sys.path.append('..')
    from config.scraping_config import DATA_PATHS
    
    structurer = DataStructurer(DATA_PATHS)
    structurer.structure_image_data()
    structurer.structure_text_data()
    structurer.create_training_dataset()
    structurer.generate_summary_report()