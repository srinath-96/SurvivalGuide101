# Gemini Fine-Tuning Guide for Survival Assistant

## 📊 Dataset Overview

Your survival data scraper has successfully collected **comprehensive, high-quality survival content** specifically designed for fine-tuning Gemini 3 as a survival assistant.

### 🎯 Data Quality Highlights

- **Relevant Content**: All text data focuses on actual survival techniques, emergency preparedness, and wildlife situations
- **Structured Format**: Ready-to-use training pairs for Gemini fine-tuning
- **Comprehensive Coverage**: 7 major survival categories with detailed content
- **Multiple Sources**: WikiHow survival guides, government emergency preparedness content, and structured Q&A

## 📁 Training Dataset Files

### Primary Training File
- **`gemini_training_dataset.json`** (532KB, 71 training examples)
  - Input/output pairs optimized for Gemini
  - Multiple training formats: tutorials, instructions, Q&A, information
  - Categories: Water, Fire, Shelter, Food, Wildlife, Navigation, Medical

### Comprehensive Source Data
- **`comprehensive_survival_data.json`** (388KB, 57 source items)
  - Raw survival content from multiple sources
  - Detailed metadata and source attribution
  - Full article content with steps, tips, and warnings

## 🔍 Content Categories

### Survival Tutorials (14 items)
- How to Survive in the Woods
- How to Build a Survival Shelter  
- How to Treat a Snake Bite
- How to Survive a Bear Attack
- How to Navigate Without a Compass
- How to Make a Survival Kit
- And more...

### Emergency Preparedness (8 items)
- Government emergency guides (Ready.gov)
- Emergency planning and kit building
- Natural disaster preparedness
- Flood, hurricane, wildfire, earthquake guides

### Survival Q&A (35 pairs)
- **Water**: Purification, sources, requirements
- **Fire**: Starting without matches, tinder selection, wet conditions
- **Shelter**: Building, location, insulation, waterproofing
- **Food**: Edible plants, fishing, insects, preservation
- **Wildlife**: Bear encounters, snake bites, animal safety
- **Navigation**: No-compass navigation, star navigation, signaling
- **Medical**: Wound treatment, hypothermia, dehydration, broken bones

## 🤖 Gemini Fine-Tuning Instructions

### 1. Data Format
The `gemini_training_dataset.json` file contains training examples in the format:
```json
{
  "input_text": "Question: How do you purify water in the wilderness?",
  "output_text": "There are several methods to purify water...",
  "category": "Water",
  "type": "qa"
}
```

### 2. Training Types
- **`qa`**: Question-answer pairs for direct queries
- **`tutorial`**: Step-by-step guides for survival tasks  
- **`instruction`**: Process-oriented instructions
- **`information`**: General survival knowledge

### 3. Recommended Fine-Tuning Approach

#### A. Data Preparation
```python
import json

# Load training data
with open('data/structured/gemini_training_dataset.json', 'r') as f:
    training_data = json.load(f)

# Format for Gemini fine-tuning
formatted_data = []
for example in training_data:
    formatted_data.append({
        "input": example["input_text"],
        "output": example["output_text"],
        "category": example["category"]
    })
```

#### B. Training Configuration
- **Dataset Size**: 71 high-quality examples
- **Categories**: 7 survival domains  
- **Recommended Epochs**: 3-5 (small dataset)
- **Learning Rate**: Conservative (1e-5 to 1e-4)
- **Validation Split**: 80/20 train/validation

#### C. Prompt Engineering
Use survival-specific system prompts:
```
You are a wilderness survival expert. Provide accurate, life-saving advice for emergency situations. Always prioritize safety and include warnings when appropriate.
```

### 4. Testing Prompts

Test your fine-tuned model with these survival scenarios:

#### Basic Survival
- "I'm lost in the forest with no supplies. What should I do first?"
- "How do I start a fire when everything is wet?"
- "What are the signs that water is safe to drink?"

#### Emergency Situations  
- "A bear is approaching my campsite. What do I do?"
- "Someone in my group has hypothermia. How do I help?"
- "We're caught in a lightning storm while hiking. Where should we go?"

#### Practical Skills
- "How do I build a shelter that will keep me warm overnight?"
- "What plants can I eat if I'm stranded?"
- "How do I signal for rescue without equipment?"

## 📈 Expected Outcomes

After fine-tuning with this dataset, your Gemini model should be able to:

✅ **Provide accurate survival advice** for wilderness emergencies
✅ **Give step-by-step instructions** for survival tasks  
✅ **Identify dangerous situations** and provide safety warnings
✅ **Answer specific questions** about water, fire, shelter, food, wildlife, navigation, and medical care
✅ **Adapt advice** to different environmental conditions
✅ **Prioritize immediate survival needs** over comfort

## 🔧 Continuous Improvement

### Expand the Dataset
To get even more training data, modify the scraper configuration:

1. **Increase article limits** in `config/scraping_config.py`
2. **Add more WikiHow articles** to the direct_articles list
3. **Run the scraper multiple times** to build a larger dataset
4. **Add specific scenarios** you want the model to handle

### Example Expansion
```python
# In run_scraper.py, increase limits:
results = orchestrator.run_full_pipeline(
    max_images=50,
    max_texts=1000  # Increase for more training data
)
```

## 🎯 Success Metrics

Monitor these metrics during and after fine-tuning:

- **Accuracy**: Correct survival advice
- **Safety**: Appropriate warnings and risk assessment  
- **Completeness**: Comprehensive responses covering all important aspects
- **Practicality**: Actionable advice for real emergency situations
- **Clarity**: Easy-to-follow instructions under stress

## 📝 Notes

- **Content Quality**: All data has been filtered for relevance and accuracy
- **Source Attribution**: Original sources preserved in metadata
- **Safety Focus**: Content emphasizes life-saving techniques
- **Practical Application**: Real-world scenarios and solutions
- **Comprehensive Coverage**: Major survival domains represented

---

**Ready for Gemini Fine-Tuning**: ✅ High-Quality Data | ✅ Proper Format | ✅ Survival-Focused | ✅ Comprehensive Coverage 