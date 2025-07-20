# Survival Data Scraper

A robust web scraping tool designed to collect survival-related images and text content for training LLMs as survival assistants.

## 🚀 Quick Start

```bash
# Clone/navigate to the repository
cd Survival_data

# Install dependencies
pip install -r requirements.txt

# Run the scraper (works without API keys!)
python run_scraper.py

# Test all components
python test_scraper.py
```

## 📁 Project Structure

```
Survival_data/
├── main.py                     # Main orchestrator script
├── run_scraper.py              # Simple run script
├── test_scraper.py             # Diagnostic test script
├── environment_setup.txt       # API keys setup instructions
├── requirements.txt            # Python dependencies
├── .gitignore                  # Git ignore patterns
├── README.md                   # This file
├── config/
│   └── scraping_config.py      # Configuration settings
├── scripts/
│   ├── fallback_image_scraper.py    # API-free image scraper
│   ├── image_scraper.py             # Enhanced scraper (with APIs)
│   ├── text_scraper.py              # Text content scraper
│   └── data_structuring.py          # Data processing utilities
├── data/
│   ├── images/                      # Downloaded images (30+ files)
│   └── structured/                  # Processed datasets
│       ├── fallback_image_metadata.json     # Image metadata
│       ├── survival_text_data.json          # Raw text data
│       └── survival_training_dataset.json   # LLM training dataset
└── logs/
    └── scraper_run.log              # Latest run logs
```

## 🔧 Features

### Image Scraping
- **API-Free Operation**: Works without API keys using Wikimedia Commons
- **Enhanced Mode**: Optional API support for Unsplash and Pexels
- **High-Quality Images**: Downloads full-resolution survival-related images
- **Legal Compliance**: Only Creative Commons and Public Domain images
- **Smart Metadata**: Captures dimensions, sources, licensing info

### Text Scraping
- **WikiHow Integration**: Scrapes survival guides and tutorials
- **Structured Extraction**: Separates titles, steps, tips, and warnings
- **Content Cleaning**: Removes ads and formatting artifacts
- **Robust Parsing**: Handles website structure changes gracefully

### Data Processing
- **Training Ready**: Formats data for LLM fine-tuning
- **Metadata Preservation**: Maintains source attribution
- **Quality Filtering**: Removes low-quality or irrelevant content
- **JSON Export**: Clean, structured output formats

## 🔑 API Keys (Optional)

For enhanced image scraping, you can add API keys:

1. **Get API Keys** (free):
   - [Unsplash Developers](https://unsplash.com/developers)
   - [Pexels API](https://www.pexels.com/api/)

2. **Create `.env` file**:
   ```bash
   UNSPLASH_ACCESS_KEY=your_key_here
   PEXELS_API_KEY=your_key_here
   ```

3. **Install python-dotenv**:
   ```bash
   pip install python-dotenv
   ```

**Note**: The scraper works perfectly without API keys using fallback methods!

## 📊 Output Data

### Image Data
- **Location**: `data/images/`
- **Format**: High-resolution JPG files
- **Metadata**: `data/structured/fallback_image_metadata.json`
- **Count**: 30+ images per run
- **Topics**: Camping, first aid, fire making, water purification, shelters, tools

### Text Data
- **Location**: `data/structured/survival_text_data.json`
- **Format**: Structured JSON with titles, content, steps, tips, warnings
- **Count**: 20-30 articles per run
- **Sources**: WikiHow, survival guides, emergency preparedness

### Training Dataset
- **Location**: `data/structured/survival_training_dataset.json`
- **Format**: Ready for LLM fine-tuning
- **Structure**: Question-answer pairs, instructions, content blocks

## 🧪 Testing

Run the diagnostic script to verify everything works:

```bash
python test_scraper.py
```

This checks:
- ✅ Dependencies installation
- ✅ Configuration loading
- ✅ Image scraper functionality
- ✅ Text scraper functionality
- ✅ Main orchestrator

## 🔧 Customization

### Modify Search Queries
Edit `config/scraping_config.py` to change:
- Image search terms
- Text source categories
- Scraping parameters
- Output directories

### Adjust Limits
In `run_scraper.py`, modify:
```python
results = orchestrator.run_full_pipeline(
    max_images=30,  # Increase for more images
    max_texts=20    # Increase for more text content
)
```

## 🛠️ Dependencies

- `requests>=2.25.1` - HTTP requests
- `beautifulsoup4>=4.9.3` - HTML parsing
- `lxml>=4.6.3` - XML/HTML parser
- `python-dotenv>=0.19.0` - Environment variables (optional)
- `Pillow>=8.3.2` - Image processing
- `tqdm>=4.62.0` - Progress bars

## 📝 License

This scraper collects data from public sources:
- **Images**: Creative Commons and Public Domain only
- **Text**: Respects robots.txt and rate limiting
- **Usage**: Educational and research purposes

## 🔍 Troubleshooting

### Common Issues

1. **"No images downloaded"**
   - Check internet connection
   - Try with API keys for more sources
   - Check logs in `logs/scraper_run.log`

2. **"Import errors"**
   - Run: `pip install -r requirements.txt`
   - Check Python version (3.7+)

3. **"Empty articles"**
   - Website structure may have changed
   - Check logs for specific errors
   - Try different categories in config

### Getting Help

1. Run diagnostics: `python test_scraper.py`
2. Check logs: `logs/scraper_run.log`
3. Verify dependencies: `pip list`

## 🎯 Use Cases

- **LLM Training**: Fine-tune models for survival assistance
- **Emergency Preparedness**: Build knowledge bases
- **Educational Content**: Create survival training materials
- **Research**: Analyze survival information patterns

---



## Model Weights:

https://huggingface.co/srinath123/gemma-3/tree/main
