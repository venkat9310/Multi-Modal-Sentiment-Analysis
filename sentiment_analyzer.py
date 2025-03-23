import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import logging

# Set up logging
logger = logging.getLogger(__name__)

# Download necessary NLTK data
try:
    nltk.download('vader_lexicon', quiet=True)
except Exception as e:
    logger.error(f"Failed to download NLTK data: {e}")

def analyze_text_sentiment(text):
    """
    Analyze sentiment in text using NLTK's SentimentIntensityAnalyzer.
    
    Args:
        text (str): The text to analyze
        
    Returns:
        tuple: (sentiment_score, detailed_scores)
            sentiment_score is a float between -1 (negative) and 1 (positive)
            detailed_scores is a dictionary with detailed sentiment scores
    """
    try:
        # Initialize the analyzer
        sia = SentimentIntensityAnalyzer()
        
        # Get sentiment scores
        sentiment_dict = sia.polarity_scores(text)
        
        # Convert compound score to range from -1 to 1
        compound_score = sentiment_dict['compound']
        
        # Create detailed breakdown
        details = {
            'positive': sentiment_dict['pos'],
            'neutral': sentiment_dict['neu'],
            'negative': sentiment_dict['neg'],
            'compound': compound_score
        }
        
        return compound_score, details
        
    except Exception as e:
        logger.exception(f"Error analyzing text sentiment: {e}")
        return None, None
