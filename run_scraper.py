#!/usr/bin/env python3
"""
Simple script to run the survival data scraper with better error handling.
"""

import os
import sys
import logging
from pathlib import Path

# Add current directory to path
sys.path.append(os.path.dirname(__file__))

# Load environment variables if .env file exists
env_file = Path('.env')
if env_file.exists():
    try:
        from dotenv import load_dotenv
        load_dotenv()
        print("✓ Loaded environment variables from .env file")
    except ImportError:
        print("⚠ python-dotenv not installed. Install with: pip install python-dotenv")
        print("⚠ You can still run without API keys using fallback methods")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/scraper_run.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main():
    """Run the survival data scraper."""
    try:
        # Import main orchestrator
        from main import SurvivalDataOrchestrator
        
        # Create orchestrator
        orchestrator = SurvivalDataOrchestrator()
        
        # Check for API keys
        has_unsplash = bool(os.getenv('UNSPLASH_ACCESS_KEY'))
        has_pexels = bool(os.getenv('PEXELS_API_KEY'))
        
        if has_unsplash or has_pexels:
            print("🔑 API keys detected - will use enhanced scraping")
            if has_unsplash:
                print("  ✓ Unsplash API key found")
            if has_pexels:
                print("  ✓ Pexels API key found")
        else:
            print("🆓 No API keys found - will use fallback scraping (free sources)")
            print("💡 For better results, add API keys to .env file (see environment_setup.txt)")
        
        print("\n🚀 Starting scraping process...")
        
        # Run the full pipeline with enhanced text scraping
        results = orchestrator.run_full_pipeline(
            max_images=30,  # Keep moderate for images
            max_texts=200   # Much more text for Gemini training
        )
        
        print("\n✅ Scraping completed successfully!")
        print(f"📊 Results summary:")
        if 'images' in results:
            print(f"  - Images: {len(results['images'])} collected")
        if 'texts' in results:
            print(f"  - Texts: {len(results['texts'])} collected")
        
        print(f"\n📁 Data saved to:")
        print(f"  - Images: data/images/")
        print(f"  - Texts: data/text/")
        print(f"  - Structured data: data/structured/")
        print(f"  - Logs: logs/")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Make sure all dependencies are installed:")
        print("   pip install -r requirements.txt")
        sys.exit(1)
        
    except Exception as e:
        logger.error(f"Scraping failed: {e}")
        print(f"❌ Scraping failed: {e}")
        print("📋 Check the logs for more details: logs/scraper_run.log")
        sys.exit(1)

if __name__ == "__main__":
    main() 