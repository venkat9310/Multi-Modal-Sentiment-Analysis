"""
Advanced Facial Expression Recognition Model
Uses DeepFace for more accurate emotion detection
"""

import os
import cv2
import logging
import numpy as np
import base64
from deepface import DeepFace
from deepface.commons import functions
import matplotlib.pyplot as plt
from io import BytesIO

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Mapping emotions to sentiment scores
EMOTION_TO_SENTIMENT = {
    'angry': -0.8,
    'disgust': -0.7,
    'fear': -0.6,
    'sad': -0.5,
    'surprise': 0.2,  # Surprise can be positive or negative
    'neutral': 0.0,
    'happy': 0.8
}

# Additional emotion mappings for better description
EMOTION_DESCRIPTIONS = {
    'angry': 'Anger',
    'disgust': 'Disgust',
    'fear': 'Fear',
    'sad': 'Sadness',
    'surprise': 'Surprise',
    'neutral': 'Neutral',
    'happy': 'Happiness'
}

# Emotion categories for grouping similar emotions
EMOTION_CATEGORIES = {
    'negative': ['angry', 'disgust', 'fear', 'sad'],
    'neutral': ['neutral', 'surprise'],
    'positive': ['happy']
}

# Define color mappings for emotions (for visualization)
EMOTION_COLORS = {
    'angry': (0, 0, 255),      # Red
    'disgust': (0, 128, 255),  # Orange
    'fear': (0, 69, 255),      # Red-Orange
    'sad': (255, 0, 0),        # Blue
    'surprise': (255, 255, 0), # Cyan
    'neutral': (128, 128, 128),# Gray
    'happy': (0, 255, 0)       # Green
}

class AdvancedFacialExpressionAnalyzer:
    """
    Advanced facial expression analyzer using DeepFace
    """
    
    def __init__(self):
        """Initialize the analyzer"""
        # Face detection model - use OpenCV's Haar Cascade for speed and reliability
        self.face_detector = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # Configure DeepFace settings
        self.emotion_model = "Emotion"  # Use DeepFace's emotion model
        self.detector_backend = "opencv"  # Use OpenCV for face detection (faster)
        self.enforce_detection = False   # Don't enforce face detection (helps with difficult images)
        
        logger.info("Advanced Facial Expression Analyzer initialized")
        
    def preprocess_image(self, image):
        """
        Preprocess the image to improve face detection
        
        Args:
            image: OpenCV image (numpy array)
            
        Returns:
            preprocessed_image: Enhanced image for better face detection
        """
        try:
            # Check if image is valid
            if image is None or image.size == 0:
                logger.error("Invalid image provided for preprocessing")
                return None
                
            # Make a copy to avoid modifying the original
            img_copy = image.copy()
            
            # Convert to grayscale if not already
            if len(img_copy.shape) == 3:
                gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
            else:
                gray = img_copy.copy()
                
            # Apply histogram equalization to improve contrast
            gray_eq = cv2.equalizeHist(gray)
            
            # Apply mild Gaussian blur to reduce noise
            gray_blur = cv2.GaussianBlur(gray_eq, (5, 5), 0)
            
            # Convert back to BGR for DeepFace if needed
            if len(img_copy.shape) == 3:
                enhanced = cv2.cvtColor(gray_blur, cv2.COLOR_GRAY2BGR)
            else:
                enhanced = gray_blur
                
            return enhanced
            
        except Exception as e:
            logger.exception(f"Error in image preprocessing: {e}")
            return image  # Return original on failure
    
    def detect_faces(self, image):
        """
        Detect faces in an image using cascade classifier for preliminary detection
        
        Args:
            image: OpenCV image (numpy array)
            
        Returns:
            list: List of detected face rectangles (x, y, w, h)
        """
        try:
            # Convert to grayscale for face detection
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            # Detect faces with cascade classifier
            faces = self.face_detector.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            if len(faces) == 0:
                logger.warning("No faces detected with cascade classifier")
                
            return faces
            
        except Exception as e:
            logger.exception(f"Error in face detection: {e}")
            return []
    
    def analyze_emotion(self, image, face_rect=None):
        """
        Analyze emotion in a face image using DeepFace
        
        Args:
            image: OpenCV image (numpy array)
            face_rect: Optional face rectangle (x, y, w, h)
            
        Returns:
            dict: Emotion analysis results or None on failure
        """
        try:
            # If face rectangle is provided, crop the face
            if face_rect is not None:
                x, y, w, h = face_rect
                # Add padding around face
                padding = int(0.2 * w)  # 20% padding
                x_pad = max(0, x - padding)
                y_pad = max(0, y - padding)
                w_pad = min(image.shape[1] - x_pad, w + 2*padding)
                h_pad = min(image.shape[0] - y_pad, h + 2*padding)
                
                face_img = image[y_pad:y_pad+h_pad, x_pad:x_pad+w_pad]
            else:
                face_img = image
                
            # Analyze emotion using DeepFace
            analysis = DeepFace.analyze(
                img_path=face_img,
                actions=['emotion'],
                enforce_detection=self.enforce_detection,
                detector_backend=self.detector_backend,
                prog_bar=False,
                silent=True
            )
            
            # Extract emotion results (DeepFace returns a list for multiple faces)
            if isinstance(analysis, list):
                if len(analysis) == 0:
                    logger.warning("DeepFace returned empty analysis")
                    return None
                emotion_result = analysis[0]
            else:
                emotion_result = analysis
                
            # Extract emotion scores
            if 'emotion' in emotion_result:
                emotions = emotion_result['emotion']
                
                # Ensure proper types
                emotions = {k: float(v) for k, v in emotions.items()}
                
                # Get dominant emotion
                dominant_emotion = max(emotions.items(), key=lambda x: x[1])[0]
                
                # Get confidence
                confidence = emotions[dominant_emotion]
                
                # Get second highest emotion
                emotions_sorted = sorted(emotions.items(), key=lambda x: x[1], reverse=True)
                second_emotion = emotions_sorted[1] if len(emotions_sorted) > 1 else None
                
                # Check for ambiguity
                is_ambiguous = False
                ambiguity_explanation = None
                
                if second_emotion and (confidence - second_emotion[1]) < 15:
                    is_ambiguous = True
                    ambiguity_explanation = f"The expression appears to be a mix of {dominant_emotion} ({confidence:.1f}%) and {second_emotion[0]} ({second_emotion[1]:.1f}%)"
                
                # Calculate weighted sentiment score
                sentiment_score = 0
                for emotion, score in emotions.items():
                    # Normalize score to 0-1 range
                    normalized_score = score / 100.0
                    # Apply emotional weight and add to sentiment
                    sentiment_score += EMOTION_TO_SENTIMENT.get(emotion, 0) * normalized_score
                    
                # Ensure score is between -1 and 1
                sentiment_score = max(-1.0, min(1.0, sentiment_score))
                
                # Return detailed analysis results
                return {
                    'emotions': emotions,
                    'dominant_emotion': dominant_emotion,
                    'confidence': confidence,
                    'sentiment_score': sentiment_score,
                    'is_ambiguous': is_ambiguous,
                    'ambiguity_explanation': ambiguity_explanation,
                    'face_rect': face_rect
                }
            else:
                logger.warning("No emotion data in DeepFace result")
                return None
                
        except Exception as e:
            logger.exception(f"Error in emotion analysis: {e}")
            return None
            
    def analyze_facial_expression(self, image):
        """
        Main function to analyze facial expression in an image
        
        Args:
            image: OpenCV image (numpy array)
            
        Returns:
            tuple: (sentiment_score, emotion_details)
                sentiment_score is a float between -1 (negative) and 1 (positive)
                emotion_details is a dictionary with detailed emotion scores and metadata
        """
        try:
            # Validate input
            if image is None or image.size == 0:
                logger.error("Invalid image provided")
                return None, None
                
            # Preprocess image
            preprocessed = self.preprocess_image(image)
            if preprocessed is None:
                preprocessed = image  # Fallback to original
            
            # Detect faces
            faces = self.detect_faces(preprocessed)
            
            if len(faces) == 0:
                # Try once more with original image if preprocessing didn't help
                faces = self.detect_faces(image)
                
                if len(faces) == 0:
                    logger.warning("No faces detected in the image")
                    
                    # Try with DeepFace directly (without face detection)
                    try:
                        full_analysis = self.analyze_emotion(image)
                        if full_analysis:
                            return full_analysis['sentiment_score'], full_analysis
                    except Exception as e:
                        logger.warning(f"DeepFace direct analysis failed: {e}")
                        
                    return None, None
            
            # Get the largest face (assuming it's the main subject)
            largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
            
            # Analyze emotion
            emotion_details = self.analyze_emotion(image, largest_face)
            
            if emotion_details is None:
                logger.warning("Emotion analysis failed")
                return None, None
                
            # Return sentiment score and details
            return emotion_details['sentiment_score'], emotion_details
            
        except Exception as e:
            logger.exception(f"Error analyzing facial expression: {e}")
            return None, None
    
    def visualize_results(self, image, emotion_details):
        """
        Create a visualization of the analyzed face with emotion labels
        
        Args:
            image: Original image
            emotion_details: Dictionary containing emotion analysis results
            
        Returns:
            base64_image: Base64 encoded image with visualizations
        """
        try:
            if emotion_details is None:
                return None
                
            # Make a copy of the image
            vis_image = image.copy()
            
            # Get face rectangle
            if 'face_rect' in emotion_details and emotion_details['face_rect'] is not None:
                x, y, w, h = emotion_details['face_rect']
                
                # Get dominant emotion and color
                dominant_emotion = emotion_details['dominant_emotion']
                color = EMOTION_COLORS.get(dominant_emotion, (255, 255, 255))
                
                # Draw face rectangle
                cv2.rectangle(vis_image, (x, y), (x+w, y+h), color, 2)
                
                # Add emotion label with confidence
                confidence = emotion_details['confidence']
                label = f"{EMOTION_DESCRIPTIONS.get(dominant_emotion, dominant_emotion)}: {confidence:.1f}%"
                cv2.putText(vis_image, label, (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                # Add ambiguity explanation if applicable
                if emotion_details['is_ambiguous'] and emotion_details['ambiguity_explanation']:
                    # Split explanation into multiple lines if needed
                    explanation = emotion_details['ambiguity_explanation']
                    y_offset = y + h + 30
                    
                    # Display the explanation
                    cv2.putText(vis_image, "Mixed Expression:", (x, y+h+20),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    cv2.putText(vis_image, explanation, (x, y_offset),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Create emotion bar chart in the corner
            if 'emotions' in emotion_details:
                self._add_emotion_bars(vis_image, emotion_details['emotions'])
            
            # Convert to RGB for displaying
            vis_image_rgb = cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB)
            
            # Convert to base64 for web display
            _, buffer = cv2.imencode('.jpg', vis_image_rgb)
            base64_image = base64.b64encode(buffer).decode('utf-8')
            
            return f"data:image/jpeg;base64,{base64_image}"
            
        except Exception as e:
            logger.exception(f"Error creating visualization: {e}")
            return None
    
    def _add_emotion_bars(self, image, emotions):
        """
        Add emotion bar charts to the bottom right of image
        
        Args:
            image: Image to add bar charts to (modified in place)
            emotions: Dictionary of emotion scores
        """
        try:
            # Sort emotions by score
            sorted_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)
            
            # Define chart dimensions
            h, w = image.shape[:2]
            chart_width = int(w * 0.25)  # 25% of image width
            chart_height = int(h * 0.3)  # 30% of image height
            
            # Define starting position (bottom right)
            start_x = w - chart_width - 20
            start_y = h - 20 - (len(sorted_emotions) * 25)
            
            # Create semi-transparent background
            overlay = image.copy()
            cv2.rectangle(overlay, (start_x - 10, start_y - 40), 
                         (start_x + chart_width + 10, h - 10), 
                         (0, 0, 0), -1)
            
            # Apply transparency
            alpha = 0.7
            cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
            
            # Add title
            cv2.putText(image, "Emotion Analysis", (start_x, start_y - 15),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Add bars for each emotion
            bar_height = 20
            max_bar_width = chart_width
            
            for i, (emotion, score) in enumerate(sorted_emotions):
                # Position for current bar
                y_pos = start_y + (i * 25)
                
                # Draw emotion label
                emotion_name = EMOTION_DESCRIPTIONS.get(emotion, emotion)
                cv2.putText(image, f"{emotion_name}", (start_x, y_pos),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Draw bar
                bar_width = int((score / 100) * max_bar_width)
                color = EMOTION_COLORS.get(emotion, (255, 255, 255))
                cv2.rectangle(image, (start_x, y_pos + 5), (start_x + bar_width, y_pos + bar_height),
                            color, -1)
                
                # Add score text
                cv2.putText(image, f"{score:.1f}%", (start_x + bar_width + 5, y_pos + 15),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                
        except Exception as e:
            logger.exception(f"Error adding emotion bars: {e}")

# Initialize the analyzer
analyzer = AdvancedFacialExpressionAnalyzer()

def analyze_facial_expression(image):
    """
    Public function to analyze facial expression using DeepFace
    
    Args:
        image: OpenCV image (numpy array)
        
    Returns:
        tuple: (sentiment_score, emotion_details)
    """
    return analyzer.analyze_facial_expression(image)

def visualize_emotion_analysis(image, emotion_details):
    """
    Public function to create a visualization of the emotion analysis
    
    Args:
        image: Original image
        emotion_details: Dictionary containing emotion analysis results
        
    Returns:
        base64_image: Base64 encoded image with visualizations
    """
    return analyzer.visualize_results(image, emotion_details)