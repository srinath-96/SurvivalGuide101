"""
Text scraping script for survival-related content from various sources.
"""

import os
import requests
import json
import time
import logging
from pathlib import Path
from typing import List, Dict, Optional
from bs4 import BeautifulSoup
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/text_scraper.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TextScraper:
    """Main class for scraping survival-related text content."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text content."""
        if not text:
            return ""
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,!?;:()\-\[\]{}"\']', '', text)
        # Remove excessive newlines
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        return text.strip()
    
    def scrape_wikihow(self, max_articles: int = 50) -> List[Dict]:
        """Scrape survival articles from WikiHow."""
        articles = []
        base_url = self.config['TEXT_SOURCES']['wikihow']['base_url']
        
        try:
            for category in self.config['TEXT_SOURCES']['wikihow']['categories']:
                if len(articles) >= max_articles:
                    break
                
                category_url = f"{base_url}/{category}"
                
                try:
                    response = self.session.get(category_url, timeout=30)
                    response.raise_for_status()
                    
                    soup = BeautifulSoup(response.content, 'html.parser')
                    
                    # Find article links
                    article_links = soup.find_all('a', href=re.compile(r'/[A-Z][a-zA-Z-]+'))
                    
                    for link in article_links:
                        if len(articles) >= max_articles:
                            break
                            
                        article_url = f"{base_url}{link.get('href', '')}"
                        
                        try:
                            article_data = self._scrape_wikihow_article(article_url)
                            if article_data:
                                articles.append(article_data)
                                logger.info(f"Scraped WikiHow article: {article_data.get('title', 'Unknown')}")
                                
                        except Exception as e:
                            logger.warning(f"Error scraping WikiHow article {article_url}: {e}")
                            
                        time.sleep(self.config['SCRAPING_PARAMS']['delay_between_requests'])
                        
                except Exception as e:
                    logger.error(f"Error accessing WikiHow category {category}: {e}")
                    
        except Exception as e:
            logger.error(f"Error scraping WikiHow: {e}")
            
        return articles
    
    def _scrape_wikihow_article(self, url: str) -> Optional[Dict]:
        """Scrape individual WikiHow article."""
        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract title - updated selectors
            title = (soup.find('h1', class_='mw-headline') or 
                    soup.find('h1') or 
                    soup.find('title'))
            title = title.text.strip() if title else url.split('/')[-1].replace('-', ' ')
            
            # Extract introduction - updated selectors
            intro = (soup.find('div', class_='mf-section-0') or 
                    soup.find('div', id='intro') or
                    soup.find('p'))
            intro_text = intro.text.strip() if intro else ""
            
            # Extract content - more robust approach
            content_text = ""
            content_div = (soup.find('div', class_='mw-parser-output') or 
                          soup.find('div', id='mw-content-text') or
                          soup.find('div', class_='content'))
            
            if content_div:
                # Get all paragraphs and lists
                paragraphs = content_div.find_all(['p', 'ol', 'ul', 'div'])
                content_parts = []
                for p in paragraphs:
                    if p.text.strip():
                        content_parts.append(self.clean_text(p.text))
                content_text = '\n\n'.join(content_parts[:10])  # Limit content
            
            # Extract steps - updated approach
            steps = []
            step_elements = (soup.find_all('div', class_='step') or
                           soup.find_all('li') or
                           soup.find_all('ol'))
            
            for i, step in enumerate(step_elements[:15]):  # Limit to 15 steps
                step_text = self.clean_text(step.text)
                if step_text and len(step_text) > 10:  # Filter out very short steps
                    steps.append({
                        'step_number': i + 1,
                        'content': step_text
                    })
            
            # Extract tips and warnings - more flexible approach
            tips = []
            warnings = []
            
            # Look for tips in various formats
            tip_elements = soup.find_all(['div', 'p', 'li'], string=re.compile(r'tip|hint|advice', re.I))
            for tip in tip_elements[:5]:  # Limit tips
                tip_text = self.clean_text(tip.text)
                if tip_text and len(tip_text) > 10:
                    tips.append(tip_text)
            
            # Look for warnings
            warning_elements = soup.find_all(['div', 'p', 'li'], string=re.compile(r'warning|caution|danger', re.I))
            for warning in warning_elements[:5]:  # Limit warnings
                warning_text = self.clean_text(warning.text)
                if warning_text and len(warning_text) > 10:
                    warnings.append(warning_text)
            
            return {
                'title': self.clean_text(title),
                'url': url,
                'source': 'wikihow',
                'introduction': self.clean_text(intro_text),
                'content': self.clean_text(content_text),
                'steps': steps,
                'tips': tips,
                'warnings': warnings,
                'scraped_at': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            
        except Exception as e:
            logger.error(f"Error scraping WikiHow article {url}: {e}")
            return None
    
    def save_text_data(self, data: List[Dict], output_file: str):
        """Save text data to JSON file."""
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved {len(data)} text items to {output_file}")
        except Exception as e:
            logger.error(f"Error saving text data: {e}")
    
    def run_scraping(self):
        """Run complete text scraping process."""
        logger.info("Starting text scraping process...")
        
        all_text_data = []
        
        # Scrape from WikiHow
        logger.info("Scraping from WikiHow...")
        wikihow_articles = self.scrape_wikihow(max_articles=30)
        all_text_data.extend(wikihow_articles)
        logger.info(f"Collected {len(wikihow_articles)} WikiHow articles")
        
        # Save all text data
        text_file = os.path.join(self.config['DATA_PATHS']['structured'], 'survival_text_data.json')
        self.save_text_data(all_text_data, text_file)
        
        logger.info(f"Text scraping complete! Collected {len(all_text_data)} text items")
        return all_text_data


if __name__ == "__main__":
    import sys
    sys.path.append('..')
    from config.scraping_config import TEXT_SOURCES, DATA_PATHS, SCRAPING_PARAMS
    
    config = {
        'TEXT_SOURCES': TEXT_SOURCES,
        'DATA_PATHS': DATA_PATHS,
        'SCRAPING_PARAMS': SCRAPING_PARAMS
    }
    
    scraper = TextScraper(config)
    scraper.run_scraping()