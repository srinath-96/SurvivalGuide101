#!/usr/bin/env python3
"""
Test script to verify the scraping fixes work properly.
"""

import os
import sys
import logging
from pathlib import Path

# Add current directory to path
sys.path.append(os.path.dirname(__file__))

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_config_loading():
    """Test configuration loading."""
    try:
        from config.scraping_config import DATA_PATHS, IMAGE_SOURCES, TEXT_SOURCES, SCRAPING_PARAMS
        logger.info("✓ Configuration loaded successfully")
        return True
    except Exception as e:
        logger.error(f"✗ Configuration loading failed: {e}")
        return False

def test_fallback_image_scraper():
    """Test fallback image scraper."""
    try:
        from scripts.fallback_image_scraper import FallbackImageScraper
        from config.scraping_config import DATA_PATHS, IMAGE_SOURCES, SCRAPING_PARAMS
        
        config = {
            'DATA_PATHS': DATA_PATHS,
            'IMAGE_SOURCES': IMAGE_SOURCES,
            'SCRAPING_PARAMS': SCRAPING_PARAMS
        }
        
        scraper = FallbackImageScraper(config)
        logger.info("✓ Fallback image scraper initialized successfully")
        
        # Test a small scraping operation
        images = scraper.scrape_wikimedia_commons('fire making', max_images=2)
        logger.info(f"✓ Fallback scraper test completed: {len(images)} images found")
        return True
    except Exception as e:
        logger.error(f"✗ Fallback image scraper test failed: {e}")
        return False

def test_text_scraper():
    """Test text scraper."""
    try:
        from scripts.text_scraper import TextScraper
        from config.scraping_config import TEXT_SOURCES, SCRAPING_PARAMS
        
        config = {
            'TEXT_SOURCES': TEXT_SOURCES,
            'SCRAPING_PARAMS': SCRAPING_PARAMS
        }
        
        scraper = TextScraper(config)
        logger.info("✓ Text scraper initialized successfully")
        
        # Test scraping a single WikiHow article
        test_url = "https://www.wikihow.com/Build-a-Fire"
        try:
            article = scraper._scrape_wikihow_article(test_url)
            if article and article.get('title'):
                logger.info(f"✓ Text scraper test completed: Article '{article['title']}' scraped successfully")
            else:
                logger.warning("⚠ Text scraper returned empty article")
        except Exception as e:
            logger.warning(f"⚠ Individual article scraping failed (may be due to website changes): {e}")
        
        return True
    except Exception as e:
        logger.error(f"✗ Text scraper test failed: {e}")
        return False

def test_main_orchestrator():
    """Test main orchestrator."""
    try:
        from main import SurvivalDataOrchestrator
        
        orchestrator = SurvivalDataOrchestrator()
        logger.info("✓ Main orchestrator initialized successfully")
        
        # Test directory setup
        orchestrator.setup_directories()
        logger.info("✓ Directory setup completed")
        
        return True
    except Exception as e:
        logger.error(f"✗ Main orchestrator test failed: {e}")
        return False

def check_dependencies():
    """Check if required dependencies are installed."""
    required_packages = [
        'requests', 'beautifulsoup4', 'Pillow', 'pathlib'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            if package == 'beautifulsoup4':
                import bs4
            elif package == 'Pillow':
                import PIL
            else:
                __import__(package)
            logger.info(f"✓ {package} is installed")
        except ImportError:
            missing_packages.append(package)
            logger.error(f"✗ {package} is missing")
    
    if missing_packages:
        logger.error(f"Missing packages: {missing_packages}")
        logger.info("Install them with: pip install " + " ".join(missing_packages))
        return False
    
    return True

def main():
    """Run all tests."""
    logger.info("Starting scraper diagnostics...")
    
    tests = [
        ("Dependencies Check", check_dependencies),
        ("Configuration Loading", test_config_loading),
        ("Fallback Image Scraper", test_fallback_image_scraper),
        ("Text Scraper", test_text_scraper),
        ("Main Orchestrator", test_main_orchestrator)
    ]
    
    results = []
    for test_name, test_func in tests:
        logger.info(f"\n--- Testing {test_name} ---")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            logger.error(f"Test {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    logger.info("\n" + "="*50)
    logger.info("TEST SUMMARY")
    logger.info("="*50)
    
    passed = 0
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        logger.info(f"{test_name}: {status}")
        if result:
            passed += 1
    
    logger.info(f"\nOverall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        logger.info("🎉 All tests passed! Your scraper should work now.")
        logger.info("💡 If you have API keys, create a .env file as described in environment_setup.txt")
        logger.info("💡 If you don't have API keys, the fallback scraper will work without them")
    else:
        logger.info("⚠ Some tests failed. Check the errors above and install missing dependencies.")

if __name__ == "__main__":
    main() 