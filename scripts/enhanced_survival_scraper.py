"""
Enhanced survival text scraper that collects high-quality survival content from multiple sources
for LLM fine-tuning. Focuses on DIY tutorials, survival tips, and Q&A for wilderness situations.
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
import random

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/enhanced_scraper.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class EnhancedSurvivalScraper:
    """Enhanced scraper for comprehensive survival content."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': config.get('SCRAPING_PARAMS', {}).get('user_agent', 
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')
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
    
    def scrape_wikihow_direct(self, max_articles: int = 100) -> List[Dict]:
        """Scrape specific WikiHow survival articles."""
        articles = []
        base_url = self.config['TEXT_SOURCES']['wikihow_direct']['base_url']
        direct_articles = self.config['TEXT_SOURCES']['wikihow_direct']['direct_articles']
        
        for i, article_slug in enumerate(direct_articles[:max_articles]):
            if i >= max_articles:
                break
                
            try:
                url = f"{base_url}/{article_slug}"
                response = self.session.get(url, timeout=30)
                response.raise_for_status()
                
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Extract title
                title_elem = (soup.find('h1', class_='mw-headline') or 
                             soup.find('h1') or 
                             soup.find('title'))
                title = title_elem.text.strip() if title_elem else article_slug.replace('-', ' ')
                
                # Extract introduction
                intro_elem = (soup.find('div', class_='mf-section-0') or 
                             soup.find('div', id='intro') or
                             soup.find('p'))
                intro_text = intro_elem.text.strip() if intro_elem else ""
                
                # Extract steps with better targeting
                steps = []
                step_elements = soup.find_all(['ol', 'div'], class_=re.compile(r'steps|method'))
                
                for step_container in step_elements:
                    step_items = step_container.find_all(['li', 'div'], class_=re.compile(r'step'))
                    for i, step in enumerate(step_items[:15]):
                        step_text = self.clean_text(step.get_text())
                        if step_text and len(step_text) > 20:
                            steps.append({
                                'step_number': i + 1,
                                'content': step_text
                            })
                
                # Extract all useful content
                content_parts = []
                for p in soup.find_all('p'):
                    text = self.clean_text(p.get_text())
                    if text and len(text) > 30:
                        content_parts.append(text)
                
                article_data = {
                    'id': f'wikihow_{hash(url) % 1000000}',
                    'title': self.clean_text(title),
                    'url': url,
                    'source': 'wikihow_survival',
                    'category': 'survival_tutorial',
                    'introduction': self.clean_text(intro_text),
                    'steps': steps,
                    'content': '\n\n'.join(content_parts[:10]),
                    'scraped_at': time.strftime('%Y-%m-%d %H:%M:%S')
                }
                
                articles.append(article_data)
                logger.info(f"Scraped WikiHow article: {title}")
                
                time.sleep(self.config['SCRAPING_PARAMS']['delay_between_requests'])
                
            except Exception as e:
                logger.warning(f"Error scraping WikiHow article {article_slug}: {e}")
                continue
        
        return articles
    
    def scrape_ready_gov(self, max_articles: int = 30) -> List[Dict]:
        """Scrape government emergency preparedness content."""
        articles = []
        base_url = self.config['TEXT_SOURCES']['ready_gov']['base_url']
        sections = self.config['TEXT_SOURCES']['ready_gov']['sections']
        emergency_guides = self.config['TEXT_SOURCES']['ready_gov']['emergency_guides']
        
        # Scrape main sections
        for section in sections:
            try:
                url = f"{base_url}/{section}"
                response = self.session.get(url, timeout=30)
                response.raise_for_status()
                
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Extract title
                title = soup.find('h1')
                title = title.text.strip() if title else section.replace('-', ' ').title()
                
                # Extract content
                content_parts = []
                for elem in soup.find_all(['p', 'li', 'div'], class_=re.compile(r'content|text|body')):
                    text = self.clean_text(elem.get_text())
                    if text and len(text) > 30:
                        content_parts.append(text)
                
                if content_parts:
                    article_data = {
                        'id': f'ready_gov_{hash(url) % 1000000}',
                        'title': f"Emergency Preparedness: {title}",
                        'url': url,
                        'source': 'ready_gov',
                        'category': 'emergency_preparedness',
                        'content': '\n\n'.join(content_parts[:15]),
                        'scraped_at': time.strftime('%Y-%m-%d %H:%M:%S')
                    }
                    
                    articles.append(article_data)
                    logger.info(f"Scraped Ready.gov section: {title}")
                
                time.sleep(self.config['SCRAPING_PARAMS']['delay_between_requests'])
                
            except Exception as e:
                logger.warning(f"Error scraping Ready.gov section {section}: {e}")
                continue
        
        # Scrape emergency guides
        for guide in emergency_guides[:max_articles - len(articles)]:
            try:
                url = f"{base_url}/{guide}"
                response = self.session.get(url, timeout=30)
                response.raise_for_status()
                
                soup = BeautifulSoup(response.content, 'html.parser')
                
                title = guide.replace('-', ' ').title() + " Emergency Guide"
                
                content_parts = []
                for elem in soup.find_all(['p', 'li', 'h2', 'h3']):
                    text = self.clean_text(elem.get_text())
                    if text and len(text) > 20:
                        content_parts.append(text)
                
                if content_parts:
                    article_data = {
                        'id': f'ready_gov_guide_{hash(url) % 1000000}',
                        'title': title,
                        'url': url,
                        'source': 'ready_gov',
                        'category': 'emergency_guide',
                        'content': '\n\n'.join(content_parts[:20]),
                        'scraped_at': time.strftime('%Y-%m-%d %H:%M:%S')
                    }
                    
                    articles.append(article_data)
                    logger.info(f"Scraped Ready.gov guide: {title}")
                
                time.sleep(self.config['SCRAPING_PARAMS']['delay_between_requests'])
                
            except Exception as e:
                logger.warning(f"Error scraping Ready.gov guide {guide}: {e}")
                continue
        
        return articles
    
    def generate_survival_qa(self) -> List[Dict]:
        """Generate structured Q&A content for survival training."""
        qa_data = []
        questions_config = self.config['QA_SOURCES']['survival_questions']
        
        # Base answers for different categories
        base_answers = {
            'Water': {
                'How do you purify water in the wilderness?': """There are several methods to purify water in the wilderness:
1. Boiling: Bring water to a rolling boil for at least 1 minute (3 minutes at high altitude)
2. Water purification tablets: Use iodine or chlorine tablets following package instructions
3. UV sterilization: Use a UV light pen if available
4. Solar disinfection: Fill clear plastic bottles and leave in direct sunlight for 6+ hours
5. Filtration: Use cloth, sand, charcoal, and gravel layers to filter, then boil
6. Distillation: Collect steam condensation from boiling water
Always let sediment settle before treating, and use the clearest water source available.""",
                
                'What are signs of contaminated water?': """Warning signs of contaminated water include:
- Unusual color (green, brown, red, or cloudy)
- Strong odors (sewage, chemical, or rotten smell)
- Visible algae or scum on surface
- Dead animals or fish nearby
- Oily or foamy appearance
- Water near industrial areas or agricultural runoff
- Stagnant water with no flow
- Water downstream from campsites or trails
Even clear, odorless water can contain harmful bacteria or parasites, so always purify water from natural sources.""",
                
                'How much water do you need per day to survive?': """Water requirements for survival:
- Minimum survival: 1 liter (1 quart) per day in temperate conditions
- Basic needs: 2-3 liters per day for sedentary activities
- Active conditions: 4-6 liters per day
- Hot climate: 6-8 liters per day
- Cold climate: 4-5 liters per day (dehydration risk often overlooked)
- Pregnancy/illness: Increase by 25-50%
Signs of dehydration: thirst, dark urine, fatigue, dizziness, dry mouth
Priority: Find water within 3 days maximum, as humans can only survive 3-7 days without water."""
            },
            'Fire': {
                'How do you start a fire without matches?': """Methods to start fire without matches:

**Friction Methods:**
1. Bow drill: Use a bow to spin a wooden spindle in a fireboard notch
2. Hand drill: Spin a wooden stick between palms in a fireboard
3. Fire plow: Push a hardwood stick along a groove in softwood

**Spark Methods:**
4. Flint and steel: Strike steel against flint to create sparks
5. Magnesium fire starter: Scrape magnesium shavings and ignite with striker
6. Battery and steel wool: Touch battery terminals to steel wool

**Solar Methods:**
7. Magnifying glass: Focus sunlight through lens onto tinder
8. Ice lens: Shape clear ice into lens to focus sunlight

**Key tips:**
- Prepare tinder nest first (dry grass, bark, paper)
- Have kindling (pencil-thick dry wood) ready
- Build fire structure before lighting
- Practice these methods before needing them""",
                
                'What materials make the best tinder?': """Best tinder materials for fire starting:

**Natural tinders:**
- Birch bark: Papery outer bark peels, burns even when wet
- Cedar bark: Fibrous inner bark, easily shredded
- Dry grass: Fine, completely dry grass bundled together
- Pine needles: Dead, brown needles from evergreen trees
- Punk wood: Soft, dry, rotted wood from dead trees
- Cattail fluff: Seed heads from cattail plants
- Fatwood: Resin-rich pine wood, burns hot and long

**Processed tinders:**
- Char cloth: Cotton fabric burned in oxygen-free container
- Steel wool: Fine grade (#0000) catches sparks easily
- Petroleum jelly cotton balls: Store-bought emergency tinder
- Dryer lint: Household lint mixed with wax
- Paper: Shredded newspaper or book pages

**Preparation tips:**
- Keep tinder completely dry
- Create a "bird's nest" shape for better airflow
- Have multiple types ready
- Practice identifying tinder in your area"""
            }
        }
        
        for category_info in questions_config:
            category = category_info['category']
            questions = category_info['questions']
            
            for question in questions:
                # Use predefined answers if available, otherwise generate basic structure
                if category in base_answers and question in base_answers[category]:
                    answer = base_answers[category][question]
                else:
                    answer = f"This is an important {category.lower()} survival question that requires specific knowledge about wilderness safety and emergency preparedness techniques."
                
                qa_item = {
                    'id': f'qa_{hash(question) % 1000000}',
                    'question': question,
                    'answer': answer,
                    'category': category,
                    'source': 'generated_qa',
                    'type': 'survival_qa',
                    'created_at': time.strftime('%Y-%m-%d %H:%M:%S')
                }
                
                qa_data.append(qa_item)
        
        logger.info(f"Generated {len(qa_data)} survival Q&A pairs")
        return qa_data
    
    def run_enhanced_scraping(self, max_total_items: int = 500) -> List[Dict]:
        """Run comprehensive survival content scraping."""
        all_content = []
        
        logger.info("Starting enhanced survival content scraping...")
        
        # 1. Scrape WikiHow survival articles
        try:
            wikihow_articles = self.scrape_wikihow_direct(max_articles=50)
            all_content.extend(wikihow_articles)
            logger.info(f"Collected {len(wikihow_articles)} WikiHow articles")
        except Exception as e:
            logger.error(f"Error scraping WikiHow: {e}")
        
        # 2. Scrape Ready.gov content
        try:
            ready_articles = self.scrape_ready_gov(max_articles=30)
            all_content.extend(ready_articles)
            logger.info(f"Collected {len(ready_articles)} Ready.gov articles")
        except Exception as e:
            logger.error(f"Error scraping Ready.gov: {e}")
        
        # 3. Generate survival Q&A
        try:
            qa_content = self.generate_survival_qa()
            all_content.extend(qa_content)
            logger.info(f"Generated {len(qa_content)} Q&A pairs")
        except Exception as e:
            logger.error(f"Error generating Q&A: {e}")
        
        # Save comprehensive dataset
        if all_content:
            output_file = Path(self.config['DATA_PATHS']['structured']) / 'comprehensive_survival_data.json'
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(all_content, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved {len(all_content)} survival content items to {output_file}")
            
            # Create training dataset
            training_data = self.create_training_dataset(all_content)
            training_file = Path(self.config['DATA_PATHS']['structured']) / 'gemini_training_dataset.json'
            
            with open(training_file, 'w', encoding='utf-8') as f:
                json.dump(training_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Created training dataset with {len(training_data)} examples")
        
        return all_content
    
    def create_training_dataset(self, content: List[Dict]) -> List[Dict]:
        """Create structured training dataset for Gemini fine-tuning."""
        training_examples = []
        
        for item in content:
            if item.get('type') == 'survival_qa':
                # Q&A format
                training_examples.append({
                    'input_text': f"Question: {item['question']}",
                    'output_text': item['answer'],
                    'category': item['category'],
                    'type': 'qa'
                })
            
            elif 'steps' in item and item['steps']:
                # Tutorial format
                steps_text = '\n'.join([f"{step['step_number']}. {step['content']}" for step in item['steps']])
                
                training_examples.append({
                    'input_text': f"How to: {item['title']}",
                    'output_text': f"{item.get('introduction', '')}\n\nSteps:\n{steps_text}",
                    'category': item.get('category', 'survival'),
                    'type': 'tutorial'
                })
                
                # Generate questions from steps
                training_examples.append({
                    'input_text': f"What are the steps to {item['title'].lower()}?",
                    'output_text': steps_text,
                    'category': item.get('category', 'survival'),
                    'type': 'instruction'
                })
            
            elif item.get('content'):
                # General content format
                training_examples.append({
                    'input_text': f"Tell me about: {item['title']}",
                    'output_text': item['content'][:2000],  # Limit length
                    'category': item.get('category', 'survival'),
                    'type': 'information'
                })
        
        return training_examples


if __name__ == "__main__":
    import sys
    sys.path.append('..')
    from config.scraping_config import TEXT_SOURCES, QA_SOURCES, DATA_PATHS, SCRAPING_PARAMS
    
    config = {
        'TEXT_SOURCES': TEXT_SOURCES,
        'QA_SOURCES': QA_SOURCES,
        'DATA_PATHS': DATA_PATHS,
        'SCRAPING_PARAMS': SCRAPING_PARAMS
    }
    
    scraper = EnhancedSurvivalScraper(config)
    content = scraper.run_enhanced_scraping(max_total_items=500)
    print(f"Collected {len(content)} survival content items") 