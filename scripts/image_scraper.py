"""
Enhanced image scraping script for survival-related images with actual content collection.
"""

import os
import requests
import json
import time
import logging
from pathlib import Path
from typing import List, Dict, Optional, Any
import hashlib
from PIL import Image
import io
import base64

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/image_scraper.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class EnhancedImageScraper:
    """Enhanced image scraper that collects actual images and their content."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
    def download_image_content(self, url: str, filename: str, output_dir: str) -> Dict[str, Any]:
        """Download actual image content and metadata."""
        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            # Save actual image file
            filepath = Path(output_dir) / filename
            with open(filepath, 'wb') as f:
                f.write(response.content)
            
            # Get image dimensions
            img = Image.open(filepath)
            width, height = img.size
            
            # Create structured data
            image_data = {
                'id': hash(url) % 1000000,
                'filename': filename,
                'original_url': url,
                'local_path': str(filepath),
                'width': width,
                'height': height,
                'format': img.format,
                'size_bytes': len(response.content),
                'metadata': {
                    'aspect_ratio': width / height,
                    'size_category': 'large' if width > 1000 else 'medium' if width > 500 else 'small'
                }
            }
            
            return image_data
            
        except Exception as e:
            logger.error(f"Failed to download image {url}: {e}")
            return None
    
    def scrape_actual_images(self, query: str, max_images: int = 50) -> List[Dict[str, Any]]:
        """Scrape actual images with content."""
        images = []
        
        # Unsplash API
        access_key = os.getenv('UNSPLASH_ACCESS_KEY')
        if access_key:
            headers = {'Authorization': f'Client-ID {access_key}'}
            
            url = f"{self.config['IMAGE_SOURCES']['unsplash']['base_url']}/search/photos"
            params = {
                'query': query,
                'per_page': 30,
                'page': 1
            }
            
            try:
                response = requests.get(url, headers=headers, params=params)
                response.raise_for_status()
                
                data = response.json()
                for photo in data.get('results', [])[:max_images]:
                    image_data = self.download_image_content(
                        photo['urls']['regular'],
                        f"unsplash_{photo['id']}.jpg",
                        self.config['DATA_PATHS']['images']
                    )
                    if image_data:
                        image_data.update({
                            'description': photo.get('description', ''),
                            'tags': [tag['title'] for tag in photo.get('tags', [])],
                            'query': query,
                            'source': 'unsplash'
                        })
                        images.append(image_data)
                        
            except Exception as e:
                logger.error(f"Error scraping Unsplash: {e}")
        
        return images
    
    def scrape_pexels_images(self, query: str, max_images: int = 50) -> List[Dict[str, Any]]:
        """Scrape actual images from Pexels."""
        images = []
        api_key = os.getenv('PEXELS_API_KEY')
        
        if api_key:
            headers = {'Authorization': api_key}
            url = f"{self.config['IMAGE_SOURCES']['pexels']['base_url']}/search"
            params = {
                'query': query,
                'per_page': 80,
                'page': 1
            }
            
            try:
                response = requests.get(url, headers=headers, params=params)
                response.raise_for_status()
                
                data = response.json()
                for photo in data.get('photos', [])[:max_images]:
                    image_data = self.download_image_content(
                        photo['src']['large'],
                        f"pexels_{photo['id']}.jpg",
                        self.config['DATA_PATHS']['images']
                    )
                    if image_data:
                        image_data.update({
                            'description': photo.get('alt', ''),
                            'photographer': photo.get('photographer', ''),
                            'source': 'pexels'
                        })
                        images.append(image_data)
                        
            except Exception as e:
                logger.error(f"Error scraping Pexels: {e}")
        
        return images
    
    def run_image_collection(self, max_images_per_source: int = 50) -> List[Dict[str, Any]]:
        """Run complete image collection with actual content."""
        all_images = []
        
        # Collect from multiple sources
        for query in self.config['IMAGE_SOURCES']['unsplash']['queries']:
            images = self.scrape_actual_images(query, max_images_per_source)
            all_images.extend(images)
            
        for query in self.config['IMAGE_SOURCES']['pexels']['queries']:
            images = self.scrape_pexels_images(query, max_images_per_source)
            all_images.extend(images)
            
        return all_images


if __name__ == "__main__":
    import sys
    sys.path.append('..')
    from config.scraping_config import IMAGE_SOURCES, DATA_PATHS, SCRAPING_PARAMS
    
    config = {
        'IMAGE_SOURCES': IMAGE_SOURCES,
        'DATA_PATHS': DATA_PATHS,
        'SCRAPING_PARAMS': SCRAPING_PARAMS
    }
    
    scraper = EnhancedImageScraper(config)
    images = scraper.run_image_collection(max_images_per_source=25)
    print(f"Collected {len(images)} actual images")