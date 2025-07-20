"""
Configuration file for survival data scraping sources and parameters.
"""

# Image scraping sources for survival scenarios
IMAGE_SOURCES = {
    'unsplash': {
        'base_url': 'https://api.unsplash.com',
        'search_endpoints': [
            '/search/photos',
            '/photos/random'
        ],
        'queries': [
            'survival camping',
            'wilderness survival',
            'emergency preparedness',
            'bushcraft',
            'outdoor survival skills',
            'survival shelter',
            'fire making',
            'water purification',
            'wilderness first aid',
            'survival tools'
        ],
        'per_page': 30,
        'max_pages': 10
    },
    'pexels': {
        'base_url': 'https://api.pexels.com/v1',
        'search_endpoint': '/search',
        'queries': [
            'survival skills',
            'camping survival',
            'emergency survival',
            'bushcraft techniques',
            'wilderness training'
        ],
        'per_page': 80,
        'max_pages': 5
    }
}

# Enhanced text scraping sources for comprehensive survival content
TEXT_SOURCES = {
    'wikihow_direct': {
        'base_url': 'https://www.wikihow.com',
        'direct_articles': [
            'Survive-in-the-Woods',
            'Build-a-Survival-Shelter',
            'Find-Water-in-the-Wild',
            'Make-Fire-Without-Matches',
            'Signal-for-Help',
            'Find-Food-in-the-Wild',
            'Treat-a-Snake-Bite',
            'Survive-a-Bear-Attack',
            'Build-a-Fire',
            'Purify-Water-in-the-Wild',
            'Navigate-Without-a-Compass',
            'Survive-Being-Lost',
            'Make-a-Survival-Kit',
            'Survive-in-the-Desert',
            'Survive-in-the-Mountains',
            'Build-a-Lean-To',
            'Catch-Fish-Without-a-Rod',
            'Start-a-Fire-in-the-Rain',
            'Survive-a-Winter-Storm',
            'Make-a-Snare',
            'Identify-Edible-Plants',
            'Treat-Hypothermia',
            'Make-a-Compass',
            'Cross-a-River-Safely',
            'Survive-a-Lightning-Storm'
        ],
        'max_articles': 100
    },
    'survival_blog': {
        'base_url': 'https://www.survivalblog.com',
        'categories': [
            'preparedness-basics',
            'survival-skills',
            'medical-preparedness',
            'self-defense',
            'food-storage'
        ],
        'max_articles': 50
    },
    'outdoor_life': {
        'base_url': 'https://www.outdoorlife.com',
        'sections': [
            'survival',
            'hunting/survival-skills',
            'fishing/survival',
            'camping'
        ],
        'max_articles': 40
    },
    'ready_gov': {
        'base_url': 'https://www.ready.gov',
        'sections': [
            'plan',
            'kit',
            'informed',
            'involved'
        ],
        'emergency_guides': [
            'floods',
            'hurricanes',
            'winter-weather',
            'wildfire',
            'earthquake',
            'severe-weather'
        ],
        'max_articles': 30
    },
    'primitive_ways': {
        'base_url': 'http://www.primitiveways.com',
        'categories': [
            'fire.html',
            'shelter.html',
            'water.html',
            'food.html',
            'tools.html'
        ],
        'max_articles': 25
    }
}

# Survival Q&A sources for training data
QA_SOURCES = {
    'survival_questions': [
        {
            'category': 'Water',
            'questions': [
                'How do you purify water in the wilderness?',
                'What are signs of contaminated water?',
                'How much water do you need per day to survive?',
                'How do you collect rainwater?',
                'What plants can provide water?'
            ]
        },
        {
            'category': 'Fire',
            'questions': [
                'How do you start a fire without matches?',
                'What materials make the best tinder?',
                'How do you start a fire in wet conditions?',
                'What is the fire triangle?',
                'How do you safely extinguish a campfire?'
            ]
        },
        {
            'category': 'Shelter',
            'questions': [
                'How do you build an emergency shelter?',
                'What makes a good shelter location?',
                'How do you insulate a shelter?',
                'How do you waterproof a shelter?',
                'What are different types of survival shelters?'
            ]
        },
        {
            'category': 'Food',
            'questions': [
                'How do you identify edible plants?',
                'How long can humans survive without food?',
                'How do you catch fish without equipment?',
                'What insects are safe to eat?',
                'How do you preserve meat without refrigeration?'
            ]
        },
        {
            'category': 'Wildlife',
            'questions': [
                'How do you survive a bear encounter?',
                'What do you do if bitten by a snake?',
                'How do you avoid dangerous wildlife?',
                'How do you treat animal bites?',
                'What sounds indicate dangerous animals nearby?'
            ]
        },
        {
            'category': 'Navigation',
            'questions': [
                'How do you navigate without a compass?',
                'How do you use stars for navigation?',
                'What are natural navigation markers?',
                'How do you signal for rescue?',
                'How do you find your way back if lost?'
            ]
        },
        {
            'category': 'Medical',
            'questions': [
                'How do you treat wounds without medical supplies?',
                'What are signs of hypothermia?',
                'How do you treat dehydration?',
                'How do you set a broken bone?',
                'What plants have medicinal properties?'
            ]
    }
    ]
}

# Data storage paths
DATA_PATHS = {
    'images': 'data/images',
    'text': 'data/text',
    'structured': 'data/structured'
}

# Scraping parameters
SCRAPING_PARAMS = {
    'delay_between_requests': 2,
    'timeout': 30,
    'retries': 3,
    'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

# API keys (to be set as environment variables)
API_KEYS = {
    'unsplash': None,  # Set UNSPLASH_ACCESS_KEY environment variable
    'pexels': None,    # Set PEXELS_API_KEY environment variable
    'reddit': None     # Set REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET
}