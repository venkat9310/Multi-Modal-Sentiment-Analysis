"""
Claude AI Integration for Advanced Sentiment Analysis
"""

import os
import sys
import base64
import logging
import json
from anthropic import Anthropic

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
# Note: the newest Anthropic model is "claude-3-5-sonnet-20241022" which was released October 22, 2024
CLAUDE_MODEL = "claude-3-5-sonnet-20241022"

def analyze_image_with_claude(image_data, prompt_text=None):
    """
    Use Claude's vision capabilities to analyze an image for sentiment and emotions.
    
    Args:
        image_data: Binary image data
        prompt_text: Optional text to include with the image analysis
        
    Returns:
        dict: Analysis results with emotions and sentiment scores
    """
    try:
        # Check for API key
        anthropic_key = os.environ.get('ANTHROPIC_API_KEY')
        if not anthropic_key:
            logger.warning("ANTHROPIC_API_KEY environment variable not set. Claude analysis unavailable.")
            return None
            
        # Initialize the client
        client = Anthropic(api_key=anthropic_key)
        
        # Encode the image as base64
        base64_image = base64.b64encode(image_data).decode('utf-8')
        
        # Construct the system prompt for consistent analysis
        system_prompt = """
        You are an expert emotion and sentiment analyst. Analyze the facial expressions and emotional cues in the image.
        Provide detailed analysis of:
        1. Primary emotion (happy, sad, angry, surprised, fearful, disgusted, neutral)
        2. Confidence level for each emotion (as percentages)
        3. Overall sentiment score on a scale from -1 (extremely negative) to 1 (extremely positive)
        
        Format your response as a JSON object with the following structure:
        {
          "dominant_emotion": "string",
          "emotions": {
            "happy": float,
            "sad": float,
            "angry": float,
            "surprise": float,
            "fear": float,
            "disgust": float,
            "neutral": float
          },
          "sentiment_score": float,
          "confidence": float,
          "analysis": "string"
        }
        Where emotions are percentages (0-100) that sum to 100.
        Only respond with the JSON object, no explanations.
        """
        
        # Construct the user message with both image and optional text
        user_message = "Analyze the emotions and sentiment in this image."
        if prompt_text:
            user_message += f" The context is: {prompt_text}"
            
        # Create the message
        message = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=1000,
            system=system_prompt,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": user_message
                        },
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": base64_image
                            }
                        }
                    ]
                }
            ]
        )
        
        # Extract the JSON response
        try:
            response_text = message.content[0].text
            # Clean the response text (remove any markdown code blocks if present)
            if "```json" in response_text:
                response_text = response_text.split("```json", 1)[1]
            if "```" in response_text:
                response_text = response_text.split("```", 1)[0]
                
            response_text = response_text.strip()
            result = json.loads(response_text)
            
            # Ensure all required fields are present
            required_fields = ['dominant_emotion', 'emotions', 'sentiment_score']
            for field in required_fields:
                if field not in result:
                    logger.warning(f"Claude response missing required field: {field}")
                    return None
                    
            # Normalize emotions to sum to 100
            total = sum(result['emotions'].values())
            if total > 0:  # Avoid division by zero
                for emotion in result['emotions']:
                    result['emotions'][emotion] = (result['emotions'][emotion] / total) * 100
                    
            # Ensure sentiment score is between -1 and 1
            result['sentiment_score'] = max(-1.0, min(1.0, result['sentiment_score']))
            
            return result
            
        except (json.JSONDecodeError, IndexError) as e:
            logger.error(f"Error parsing Claude response: {e}")
            logger.debug(f"Raw response: {message.content}")
            return None
            
    except Exception as e:
        logger.exception(f"Error analyzing image with Claude: {e}")
        return None