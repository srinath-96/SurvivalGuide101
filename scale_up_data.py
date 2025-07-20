#!/usr/bin/env python3
"""
Scale-up script to collect massive amounts of survival data for Gemini fine-tuning.
This script can be run multiple times to build a comprehensive survival dataset.
"""

import os
import sys
import json
import logging
from pathlib import Path

sys.path.append(os.path.dirname(__file__))

from scripts.enhanced_survival_scraper import EnhancedSurvivalScraper
from config.scraping_config import TEXT_SOURCES, QA_SOURCES, DATA_PATHS, SCRAPING_PARAMS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def scale_up_survival_data(target_size: int = 1000):
    """Scale up survival data collection to target size."""
    
    print(f"🎯 Target: {target_size} training examples for Gemini fine-tuning")
    print("🚀 Starting enhanced data collection...")
    
    config = {
        'TEXT_SOURCES': TEXT_SOURCES,
        'QA_SOURCES': QA_SOURCES,
        'DATA_PATHS': DATA_PATHS,
        'SCRAPING_PARAMS': SCRAPING_PARAMS
    }
    
    # Collect from all sources
    scraper = EnhancedSurvivalScraper(config)
    
    # Scale up each source
    all_content = []
    
    # 1. Enhanced WikiHow collection
    print("📚 Collecting WikiHow survival articles...")
    wikihow_articles = scraper.scrape_wikihow_direct(max_articles=100)
    all_content.extend(wikihow_articles)
    print(f"  ✅ Collected {len(wikihow_articles)} WikiHow articles")
    
    # 2. Government sources  
    print("🏛️ Collecting government emergency guides...")
    ready_articles = scraper.scrape_ready_gov(max_articles=50)
    all_content.extend(ready_articles)
    print(f"  ✅ Collected {len(ready_articles)} government guides")
    
    # 3. Enhanced Q&A generation
    print("💬 Generating comprehensive Q&A pairs...")
    qa_content = scraper.generate_survival_qa()
    all_content.extend(qa_content)
    print(f"  ✅ Generated {len(qa_content)} Q&A pairs")
    
    # 4. Create expanded training dataset
    print("🤖 Creating Gemini training dataset...")
    training_data = scraper.create_training_dataset(all_content)
    
    # Save comprehensive dataset
    output_file = Path(DATA_PATHS['structured']) / f'large_survival_dataset_{len(all_content)}_items.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_content, f, indent=2, ensure_ascii=False)
    
    # Save training dataset
    training_file = Path(DATA_PATHS['structured']) / f'gemini_large_training_{len(training_data)}_examples.json'
    with open(training_file, 'w', encoding='utf-8') as f:
        json.dump(training_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n🎉 COLLECTION COMPLETE!")
    print(f"📊 Results:")
    print(f"  • Total source items: {len(all_content)}")
    print(f"  • Training examples: {len(training_data)}")
    print(f"  • Files saved:")
    print(f"    - {output_file}")
    print(f"    - {training_file}")
    
    # Show breakdown
    categories = {}
    for item in all_content:
        cat = item.get('category', 'unknown')
        categories[cat] = categories.get(cat, 0) + 1
    
    print(f"\n📈 Content Breakdown:")
    for cat, count in sorted(categories.items()):
        print(f"  • {cat}: {count} items")
    
    # Recommendations for scaling further
    if len(training_data) < target_size:
        print(f"\n💡 To reach {target_size} examples:")
        print(f"  1. Add more WikiHow articles to config/scraping_config.py")
        print(f"  2. Expand the Q&A categories and questions")
        print(f"  3. Run this script multiple times with different configurations")
        print(f"  4. Add more government and educational sources")
    
    return len(training_data)

def add_more_wikihow_articles():
    """Suggest additional WikiHow articles for manual addition."""
    additional_articles = [
        'Survive-a-Tornado',
        'Survive-an-Earthquake',
        'Survive-a-Tsunami', 
        'Build-a-Fire-Pit',
        'Make-Emergency-Candles',
        'Preserve-Food-Without-Refrigeration',
        'Find-North-Without-a-Compass',
        'Tie-Survival-Knots',
        'Build-a-Water-Filter',
        'Make-a-Solar-Still',
        'Treat-Burns-in-the-Wild',
        'Set-a-Broken-Arm',
        'Recognize-Poisonous-Plants',
        'Make-Natural-Insect-Repellent',
        'Build-a-Root-Cellar',
        'Preserve-Meat-by-Smoking',
        'Make-Survival-Tools-from-Stone',
        'Build-a-Raft',
        'Survive-in-the-Arctic',
        'Survive-in-a-Swamp',
        'Make-Emergency-Soap',
        'Tan-Animal-Hides',
        'Make-Primitive-Weapons',
        'Build-Underground-Shelter',
        'Create-Natural-Medicine'
    ]
    
    print(f"\n📝 SUGGESTED ADDITIONAL WIKIHOW ARTICLES:")
    print("Add these to config/scraping_config.py -> direct_articles:")
    for article in additional_articles:
        print(f"  '{article}',")
    
    return additional_articles

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Scale up survival data collection')
    parser.add_argument('--target', type=int, default=1000, help='Target number of training examples')
    parser.add_argument('--suggestions', action='store_true', help='Show suggestions for more content')
    
    args = parser.parse_args()
    
    if args.suggestions:
        add_more_wikihow_articles()
    else:
        examples_collected = scale_up_survival_data(args.target)
        
        if examples_collected >= args.target:
            print(f"🎯 TARGET REACHED! You now have {examples_collected} training examples.")
        else:
            print(f"📈 Progress: {examples_collected}/{args.target} examples collected.")
            print("Run with --suggestions to see how to add more content.") 