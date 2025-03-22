import re
import string
import logging
import nltk

# Download necessary NLTK data
try:
    nltk.data.find('stopwords')
    nltk.data.find('punkt')
except LookupError:
    nltk.download('stopwords')
    nltk.download('punkt')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

logger = logging.getLogger(__name__)

def preprocess_text(text):
    """
    Preprocess the text for sentiment analysis
    
    Args:
        text: Input text string
        
    Returns:
        Preprocessed text
    """
    try:
        if not text or text.strip() == '':
            return ''
            
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        
        # Remove mentions and hashtags (for social media text)
        text = re.sub(r'@\w+|#\w+', '', text)
        
        # Remove punctuation that's not part of emoticons
        # Keep basic emoticons like :) :( :D
        text = re.sub(r'[^\w\s:;)(><-]', '', text)
        
        # Keep emoticons intact as they're important for sentiment
        emoticons = {
            ':)': ' happy_emoji ',
            ':(': ' sad_emoji ',
            ':D': ' very_happy_emoji ',
            ':P': ' playful_emoji ',
            ':/': ' confused_emoji ',
            ':o': ' surprised_emoji ',
            ';)': ' wink_emoji '
        }
        
        for emoticon, replacement in emoticons.items():
            text = text.replace(emoticon, replacement)
            
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords (but keep negation words like "not", "no", etc.)
        stop_words = set(stopwords.words('english')) - {'no', 'not', 'nor', 'none', 'never', 'neither'}
        tokens = [word for word in tokens if word not in stop_words]
        
        # Join tokens back into a string
        processed_text = ' '.join(tokens)
        
        return processed_text
        
    except Exception as e:
        logger.error(f"Error preprocessing text: {str(e)}")
        return text  # Return original text if preprocessing fails
