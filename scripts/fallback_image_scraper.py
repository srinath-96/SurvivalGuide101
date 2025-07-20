"""
Fallback image scraper that works without API keys using free sources.
"""

import os
import requests
import json
import time
import logging
from pathlib import Path
from typing import List, Dict, Optional
import hashlib
from PIL import Image
import io
from bs4 import BeautifulSoup
import urllib.parse

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/image_scraper.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class FallbackImageScraper:
    """Fallback image scraper that doesn't require API keys."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
    def download_image(self, url: str, filename: str, output_dir: str, metadata: Dict = None) -> Optional[Dict]:
        """Download image and return metadata."""
        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            # Validate content type
            content_type = response.headers.get('content-type', '')
            if not content_type.startswith('image/'):
                logger.warning(f"Invalid content type {content_type} for {url}")
                return None
            
            # Create output directory if it doesn't exist
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            
            # Save image
            filepath = Path(output_dir) / filename
            with open(filepath, 'wb') as f:
                f.write(response.content)
            
            # Get image info
            try:
                img = Image.open(filepath)
                width, height = img.size
                format_type = img.format
            except Exception as e:
                logger.warning(f"Could not read image info for {filepath}: {e}")
                width, height, format_type = 0, 0, 'unknown'
            
            image_data = {
                'id': hashlib.md5(url.encode()).hexdigest()[:8],
                'filename': filename,
                'original_url': url,
                'local_path': str(filepath),
                'width': width,
                'height': height,
                'format': format_type,
                'size_bytes': len(response.content),
                'source': 'fallback_scraper'
            }
            
            if metadata:
                image_data.update(metadata)
            
            logger.info(f"Downloaded image: {filename} ({width}x{height})")
            return image_data
            
        except Exception as e:
            logger.error(f"Failed to download image {url}: {e}")
            return None
    
    def scrape_wikimedia_commons(self, query: str, max_images: int = 20) -> List[Dict]:
        """Scrape images from Wikimedia Commons (free to use)."""
        images = []
        
        try:
            # Search Wikimedia Commons
            search_url = "https://commons.wikimedia.org/w/api.php"
            params = {
                'action': 'query',
                'format': 'json',
                'list': 'search',
                'srsearch': f'filetype:bitmap {query}',
                'srnamespace': 6,  # File namespace
                'srlimit': max_images
            }
            
            response = self.session.get(search_url, params=params)
            response.raise_for_status()
            data = response.json()
            
            for item in data.get('query', {}).get('search', []):
                title = item['title']
                # Get image info
                info_params = {
                    'action': 'query',
                    'format': 'json',
                    'titles': title,
                    'prop': 'imageinfo',
                    'iiprop': 'url|size|metadata'
                }
                
                try:
                    info_response = self.session.get(search_url, params=info_params)
                    info_data = info_response.json()
                    
                    pages = info_data.get('query', {}).get('pages', {})
                    for page_id, page_data in pages.items():
                        imageinfo = page_data.get('imageinfo', [])
                        if imageinfo:
                            img_url = imageinfo[0].get('url')
                            if img_url:
                                filename = f"wikimedia_{page_id}_{len(images)}.jpg"
                                metadata = {
                                    'title': title,
                                    'query': query,
                                    'source': 'wikimedia_commons',
                                    'license': 'Creative Commons'
                                }
                                
                                image_data = self.download_image(
                                    img_url, filename, 
                                    self.config.get('DATA_PATHS', {}).get('images', 'data/images'),
                                    metadata
                                )
                                
                                if image_data:
                                    images.append(image_data)
                                    
                                time.sleep(1)  # Be respectful to Wikimedia
                                
                except Exception as e:
                    logger.warning(f"Error getting image info for {title}: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"Error scraping Wikimedia Commons: {e}")
            
        return images
    
    def scrape_government_sources(self, max_images: int = 15) -> List[Dict]:
        """Scrape from government sources (public domain)."""
        images = []
        
        # Example: FEMA emergency preparedness images (public domain)
        gov_urls = [
            "https://www.ready.gov/sites/default/files/2019-04/emergency-supply-kit.jpg",
            "https://www.ready.gov/sites/default/files/2019-04/fire-extinguisher.jpg",
            # Add more known government emergency preparedness images
        ]
        
        for i, url in enumerate(gov_urls[:max_images]):
            try:
                filename = f"gov_emergency_{i}.jpg"
                metadata = {
                    'source': 'government',
                    'license': 'Public Domain',
                    'category': 'emergency_preparedness'
                }
                
                image_data = self.download_image(
                    url, filename,
                    self.config.get('DATA_PATHS', {}).get('images', 'data/images'),
                    metadata
                )
                
                if image_data:
                    images.append(image_data)
                    
                time.sleep(2)  # Be respectful
                
            except Exception as e:
                logger.warning(f"Error downloading government image {url}: {e}")
                
        return images
    
    def run_fallback_scraping(self, max_images_per_source: int = 20) -> List[Dict]:
        """Run fallback image scraping from free sources."""
        all_images = []
        
        logger.info("Running fallback image scraping (no API keys required)...")
        
        # Survival-related queries for Wikimedia
        queries = [
            'survival camping',
            'wilderness first aid',
            'fire making',
            'water purification',
            'emergency shelter',
            'survival tools'
        ]
        
        # Scrape from Wikimedia Commons
        for query in queries:
            try:
                images = self.scrape_wikimedia_commons(query, max_images_per_source // len(queries))
                all_images.extend(images)
                logger.info(f"Found {len(images)} images for query: {query}")
                time.sleep(3)  # Rate limiting
            except Exception as e:
                logger.error(f"Error scraping Wikimedia for {query}: {e}")
        
        # Scrape from government sources
        try:
            gov_images = self.scrape_government_sources(10)
            all_images.extend(gov_images)
            logger.info(f"Found {len(gov_images)} government images")
        except Exception as e:
            logger.error(f"Error scraping government sources: {e}")
        
        # Save metadata
        if all_images:
            metadata_file = Path(self.config.get('DATA_PATHS', {}).get('structured', 'data/structured')) / 'fallback_image_metadata.json'
            metadata_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(metadata_file, 'w') as f:
                json.dump(all_images, f, indent=2)
            
            logger.info(f"Saved metadata for {len(all_images)} images to {metadata_file}")
        
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
    
    scraper = FallbackImageScraper(config)
    images = scraper.run_fallback_scraping(max_images_per_source=10)
    print(f"Collected {len(images)} images using fallback methods") 