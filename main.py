#!/usr/bin/env python3
"""
Main orchestration script for survival data scraping and structuring.
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Add scripts directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts'))

from config.scraping_config import DATA_PATHS, IMAGE_SOURCES, TEXT_SOURCES, QA_SOURCES, SCRAPING_PARAMS
from scripts.image_scraper import EnhancedImageScraper
from scripts.fallback_image_scraper import FallbackImageScraper
from scripts.text_scraper import TextScraper
from scripts.enhanced_survival_scraper import EnhancedSurvivalScraper
from scripts.data_structuring import DataStructurer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/main.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SurvivalDataOrchestrator:
    """Main orchestrator for survival data collection and processing."""
    
    def __init__(self):
        self.config = {
            'IMAGE_SOURCES': IMAGE_SOURCES,
            'TEXT_SOURCES': TEXT_SOURCES,
            'QA_SOURCES': QA_SOURCES,
            'DATA_PATHS': DATA_PATHS,
            'SCRAPING_PARAMS': SCRAPING_PARAMS
        }
        
    def setup_directories(self):
        """Create necessary directories."""
        directories = [
            'data/images',
            'data/text',
            'data/structured',
            'logs',
            'config',
            'scripts'
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory: {directory}")
    
    def run_image_scraping(self, max_images_per_source: int = 50):
        """Run image scraping process with fallback options."""
        logger.info("Starting image scraping...")
        
        # Check if API keys are available
        has_unsplash = bool(os.getenv('UNSPLASH_ACCESS_KEY'))
        has_pexels = bool(os.getenv('PEXELS_API_KEY'))
        
        images = []
        
        if has_unsplash or has_pexels:
            logger.info("API keys found, using enhanced scraper...")
            scraper = EnhancedImageScraper(self.config)
            try:
                images = scraper.run_image_collection(max_images_per_source=max_images_per_source)
            except Exception as e:
                logger.error(f"Enhanced scraper failed: {e}")
                images = []
        
        # If no API keys or enhanced scraper failed, use fallback
        if not images:
            logger.info("Using fallback scraper (no API keys required)...")
            fallback_scraper = FallbackImageScraper(self.config)
            images = fallback_scraper.run_fallback_scraping(max_images_per_source=max_images_per_source)
        
        logger.info(f"Image scraping completed: {len(images)} images collected")
        return images
    
    def run_text_scraping(self, max_texts_per_source: int = 500):
        """Run enhanced text scraping process for comprehensive survival content."""
        logger.info("Starting enhanced survival text scraping...")
        
        # Use enhanced scraper for better survival content
        enhanced_scraper = EnhancedSurvivalScraper(self.config)
        texts = enhanced_scraper.run_enhanced_scraping(max_total_items=max_texts_per_source)
        
        # Also run basic scraper as backup if needed
        if len(texts) < 50:
            logger.info("Running backup text scraper...")
            backup_scraper = TextScraper(self.config)
            backup_texts = backup_scraper.run_scraping()
            texts.extend(backup_texts)
        
        logger.info(f"Enhanced text scraping completed: {len(texts)} items collected")
        return texts
    
    def run_data_structuring(self):
        """Run data structuring process."""
        logger.info("Starting data structuring...")
        structurer = DataStructurer(self.config['DATA_PATHS'])
        
        # Structure image data
        image_data = structurer.structure_image_data()
        
        # Structure text data
        text_data = structurer.structure_text_data()
        
        # Create training dataset
        training_data = structurer.create_training_dataset()
        
        logger.info("Data structuring completed")
        return {
            'images': image_data,
            'texts': text_data,
            'training_dataset': training_data
        }
    
    def run_full_pipeline(self, max_images: int = 50, max_texts: int = 500):
        """Run complete data collection and processing pipeline."""
        logger.info("Starting full pipeline...")
        
        # Setup directories
        self.setup_directories()
        
        # Run scraping
        images = self.run_image_scraping(max_images)
        texts = self.run_text_scraping(max_texts)
        
        # Run structuring
        structured_data = self.run_data_structuring()
        
        # Generate summary
        summary = structured_data['training_dataset']
        
        logger.info("Full pipeline completed successfully!")
        return {
            'images_collected': len(images),
            'texts_collected': len(texts),
            'summary': summary
        }
    
    def validate_data(self):
        """Validate collected data."""
        logger.info("Validating collected data...")
        
        validation_results = {
            'images': {
                'total': 0,
                'valid': 0,
                'invalid': 0,
                'errors': []
            },
            'texts': {
                'total': 0,
                'valid': 0,
                'invalid': 0,
                'errors': []
            }
        }
        
        # Validate images
        images_file = Path(self.config['DATA_PATHS']['structured']) / 'structured_images.json'
        if images_file.exists():
            with open(images_file, 'r') as f:
                images = json.load(f)
            
            validation_results['images']['total'] = len(images)
            for img in images:
                if img.get('id') and img.get('url') and img.get('category'):
                    validation_results['images']['valid'] += 1
                else:
                    validation_results['images']['invalid'] += 1
                    validation_results['images']['errors'].append(f"Invalid image: {img.get('id', 'unknown')}")
        
        # Validate texts
        texts_file = Path(self.config['DATA_PATHS']['structured']) / 'structured_texts.json'
        if texts_file.exists():
            with open(texts_file, 'r') as f:
                texts = json.load(f)
            
            validation_results['texts']['total'] = len(texts)
            for text in texts:
                if text.get('title') and text.get('content') and text.get('category'):
                    validation_results['texts']['valid'] += 1
                else:
                    validation_results['texts']['invalid'] += 1
                    validation_results['texts']['errors'].append(f"Invalid text: {text.get('title', 'unknown')}")
        
        logger.info(f"Validation completed: {validation_results}")
        return validation_results


def main():
    """Main function with CLI interface."""
    parser = argparse.ArgumentParser(description='Survival Data Collection and Processing')
    parser.add_argument('--mode', choices=['images', 'texts', 'structure', 'full', 'validate'], 
                       default='full', help='Operation mode')
    parser.add_argument('--max-images', type=int, default=50, help='Max images per source')
    parser.add_argument('--max-texts', type=int, default=30, help='Max texts per source')
    parser.add_argument('--validate', action='store_true', help='Validate collected data')
    
    args = parser.parse_args()
    
    orchestrator = SurvivalDataOrchestrator()
    
    if args.mode == 'images':
        orchestrator.run_image_scraping(args.max_images)
    elif args.mode == 'texts':
        orchestrator.run_text_scraping(args.max_texts)
    elif args.mode == 'structure':
        orchestrator.run_data_structuring()
    elif args.mode == 'full':
        result = orchestrator.run_full_pipeline(args.max_images, args.max_texts)
        print(f"\nPipeline completed:")
        print(f"Images collected: {result['images_collected']}")
        print(f"Texts collected: {result['texts_collected']}")
        print(f"Training dataset created: {result['summary']['metadata']['total_images']} images, "
              f"{result['summary']['metadata']['total_texts']} texts")
    elif args.mode == 'validate':
        validation = orchestrator.validate_data()
        print(json.dumps(validation, indent=2))
    
    if args.validate:
        validation = orchestrator.validate_data()
        print("\nValidation Results:")
        print(json.dumps(validation, indent=2))


if __name__ == "__main__":
    main()