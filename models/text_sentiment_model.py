import re
import logging
from textblob import TextBlob
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Download necessary NLTK data
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

logger = logging.getLogger(__name__)

class TextSentimentAnalyzer:
    """Class for text sentiment analysis using NLTK and TextBlob"""
    
    def __init__(self):
        """Initialize the sentiment analyzers"""
        try:
            self.vader = SentimentIntensityAnalyzer()
            logger.info("Text sentiment analyzer initialized")
        except Exception as e:
            logger.error(f"Error initializing sentiment analyzer: {str(e)}")
            raise
    
    def analyze(self, text):
        """Analyze the sentiment of the provided text"""
        if not text or text.strip() == '':
            return {'score': 0, 'label': 'neutral', 'probabilities': {'positive': 0.33, 'neutral': 0.34, 'negative': 0.33}}
        
        try:
            # Get sentiment scores using both methods
            vader_scores = self.vader.polarity_scores(text)
            textblob_analysis = TextBlob(text)
            textblob_score = textblob_analysis.sentiment.polarity
            
            # Combine scores (weighted average)
            # VADER is better for social media text and short statements
            # TextBlob provides a simpler polarity score
            combined_score = vader_scores['compound'] * 0.7 + textblob_score * 0.3
            
            # Determine sentiment label
            if combined_score >= 0.05:
                sentiment_label = 'positive'
            elif combined_score <= -0.05:
                sentiment_label = 'negative'
            else:
                sentiment_label = 'neutral'
            
            # Calculate probabilities for each sentiment class
            # Map VADER scores to probability distribution
            pos_prob = (vader_scores['pos'] + max(0, textblob_score)) / 2
            neg_prob = (vader_scores['neg'] + max(0, -textblob_score)) / 2
            neu_prob = vader_scores['neu'] * (1 - abs(textblob_score))
            
            # Normalize probabilities to sum to 1
            total = pos_prob + neg_prob + neu_prob
            if total > 0:
                pos_prob /= total
                neg_prob /= total
                neu_prob /= total
            else:
                pos_prob = neg_prob = neu_prob = 1/3
            
            return {
                'score': round(combined_score, 2),
                'label': sentiment_label,
                'probabilities': {
                    'positive': round(pos_prob, 2),
                    'neutral': round(neu_prob, 2),
                    'negative': round(neg_prob, 2)
                },
                'text': text[:100] + ('...' if len(text) > 100 else '')
            }
            
        except Exception as e:
            logger.error(f"Error in text sentiment analysis: {str(e)}")
            return {'score': 0, 'label': 'neutral', 'probabilities': {'positive': 0.33, 'neutral': 0.34, 'negative': 0.33}, 'error': str(e)}
