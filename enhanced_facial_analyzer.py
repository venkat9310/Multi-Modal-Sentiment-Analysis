"""
Enhanced Facial Expression Recognition
Uses a hybrid approach for better emotion detection
"""

import os
import cv2
import logging
import numpy as np
import base64
from collections import defaultdict
import matplotlib.pyplot as plt
from io import BytesIO

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants for facial expression recognition
EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

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

# Load facial landmark detector and face detector
FACE_CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
EYE_CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_eye.xml'
SMILE_CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_smile.xml'

class EnhancedFacialExpressionAnalyzer:
    """
    Enhanced facial expression analyzer using hybrid approach
    """
    
    def __init__(self):
        """Initialize the analyzer"""
        # Load face and feature detectors
        self.face_detector = cv2.CascadeClassifier(FACE_CASCADE_PATH)
        self.eye_detector = cv2.CascadeClassifier(EYE_CASCADE_PATH)
        self.smile_detector = cv2.CascadeClassifier(SMILE_CASCADE_PATH)
        
        logger.info("Enhanced Facial Expression Analyzer initialized")
    
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
            
            return gray_blur
            
        except Exception as e:
            logger.exception(f"Error in image preprocessing: {e}")
            return None
    
    def detect_faces(self, image):
        """
        Detect faces in an image using cascade classifier
        
        Args:
            image: OpenCV image (numpy array)
            
        Returns:
            list: List of detected face rectangles (x, y, w, h)
        """
        try:
            # Convert to grayscale for face detection if not already
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            # Use multiple scale factors for better detection
            scale_factors = [1.1, 1.15, 1.2]
            min_neighbors_options = [3, 4, 5]
            
            all_faces = []
            
            # Try different parameters for more robust detection
            for scale in scale_factors:
                for min_neighbors in min_neighbors_options:
                    faces = self.face_detector.detectMultiScale(
                        gray,
                        scaleFactor=scale,
                        minNeighbors=min_neighbors,
                        minSize=(30, 30),
                        flags=cv2.CASCADE_SCALE_IMAGE
                    )
                    
                    if len(faces) > 0:
                        all_faces.extend(faces)
                        
            # If we found faces, return them
            if len(all_faces) > 0:
                # Convert to numpy array for easier manipulation
                return np.array(all_faces)
            
            # If no faces found, try once more with more aggressive params
            return self.face_detector.detectMultiScale(
                gray,
                scaleFactor=1.05,
                minNeighbors=2,
                minSize=(20, 20),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
        except Exception as e:
            logger.exception(f"Error in face detection: {e}")
            return []
    
    def extract_facial_features(self, image, face):
        """
        Extract facial features from a detected face
        
        Args:
            image: Original image
            face: Face rectangle (x, y, w, h)
            
        Returns:
            dict: Dictionary of facial features
        """
        try:
            x, y, w, h = face
            
            # Extract the face ROI
            face_roi = image[y:y+h, x:x+w]
            
            # Convert to grayscale if needed
            if len(face_roi.shape) == 3:
                gray_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            else:
                gray_roi = face_roi
            
            # Feature dictionary
            features = {}
            
            # Detect eyes in the face
            eyes = self.eye_detector.detectMultiScale(gray_roi)
            features['eye_count'] = len(eyes)
            
            # Detect smile in the face
            smile = self.smile_detector.detectMultiScale(
                gray_roi, 
                scaleFactor=1.7, 
                minNeighbors=22, 
                minSize=(25, 25)
            )
            features['has_smile'] = len(smile) > 0
            
            # Calculate proportions and regions
            features['face_proportion'] = (w * h) / (image.shape[0] * image.shape[1])
            
            # Analyze facial regions
            # Upper face (forehead and eyes) - typically where surprise, anger show
            upper_face = gray_roi[0:int(h*0.5), :]
            # Middle face (nose, cheeks) - neutral expressions
            middle_face = gray_roi[int(h*0.3):int(h*0.7), :]
            # Lower face (mouth, chin) - happiness, sadness
            lower_face = gray_roi[int(h*0.5):h, :]
            
            # Calculate variances in different regions
            # Higher variance often means more expression in that region
            features['upper_variance'] = np.var(upper_face) if upper_face.size > 0 else 0
            features['middle_variance'] = np.var(middle_face) if middle_face.size > 0 else 0
            features['lower_variance'] = np.var(lower_face) if lower_face.size > 0 else 0
            
            # Measure overall contrast
            features['contrast'] = np.std(gray_roi) if gray_roi.size > 0 else 0
            
            return features
            
        except Exception as e:
            logger.exception(f"Error extracting facial features: {e}")
            return {}
    
    def analyze_emotion_from_features(self, features):
        """
        Analyze emotion based on extracted facial features
        
        Args:
            features: Dictionary of facial features
            
        Returns:
            dict: Dictionary of emotion probabilities
        """
        try:
            if not features:
                return None
                
            # Initialize emotions with base values
            emotions = {
                'angry': 5.0,
                'disgust': 5.0,
                'fear': 5.0,
                'happy': 5.0,
                'sad': 5.0,
                'surprise': 5.0,
                'neutral': 30.0  # Neutral starts with a higher baseline
            }
            
            # Apply rules based on features
            # Smile is a strong indicator of happiness
            if features.get('has_smile', False):
                emotions['happy'] += 55.0
                emotions['sad'] -= 10.0
                emotions['angry'] -= 10.0
                emotions['disgust'] -= 10.0
                emotions['neutral'] -= 10.0
            else:
                emotions['happy'] -= 15.0
                
            # Eye count can indicate surprise (wide eyes) or fear
            if features.get('eye_count', 0) >= 2:
                emotions['surprise'] += 5.0
                emotions['fear'] += 3.0
            else:
                emotions['surprise'] -= 5.0
                
            # Variance in upper face (forehead, eyes) - high variance for surprise, anger
            upper_variance = features.get('upper_variance', 0)
            if upper_variance > 1000:
                emotions['surprise'] += 20.0
                emotions['angry'] += 15.0
                emotions['neutral'] -= 10.0
            elif upper_variance < 500:
                emotions['neutral'] += 10.0
                emotions['surprise'] -= 5.0
                
            # Variance in lower face (mouth) - high for happiness, sadness
            lower_variance = features.get('lower_variance', 0)
            if lower_variance > 1000:
                emotions['happy'] += 15.0
                emotions['sad'] += 10.0
                emotions['neutral'] -= 10.0
            elif lower_variance < 500:
                emotions['neutral'] += 15.0
                
            # Contrast can indicate emotional intensity
            contrast = features.get('contrast', 0)
            if contrast > 60:
                # High contrast often means stronger emotions
                emotions['neutral'] -= 15.0
                emotions['happy'] += 5.0
                emotions['angry'] += 5.0
                emotions['surprise'] += 5.0
            else:
                # Lower contrast often means more neutral
                emotions['neutral'] += 15.0
                
            # If has smile and high lower variance, very likely happy
            if features.get('has_smile', False) and lower_variance > 800:
                emotions['happy'] += 20.0
                
            # If high upper variance and no smile, likely angry or surprised
            if upper_variance > 1000 and not features.get('has_smile', False):
                emotions['angry'] += 15.0
                emotions['surprise'] += 15.0
                emotions['fear'] += 10.0
                
            # Ensure emotions are in the correct range (0-100)
            for emotion in emotions:
                emotions[emotion] = max(0, min(100, emotions[emotion]))
                
            # Normalize to ensure sum is 100%
            total = sum(emotions.values())
            if total > 0:
                for emotion in emotions:
                    emotions[emotion] = (emotions[emotion] / total) * 100
                    
            return emotions
            
        except Exception as e:
            logger.exception(f"Error analyzing emotions from features: {e}")
            return {emotion: 100/len(EMOTIONS) for emotion in EMOTIONS}  # Equal distribution on error
    
    def detect_emotion_confidence(self, emotions):
        """
        Calculate confidence score and determine if emotion is ambiguous
        
        Args:
            emotions: Dictionary of emotion probabilities
            
        Returns:
            tuple: (confidence, is_ambiguous, explanation)
        """
        try:
            # Sort emotions by probability
            sorted_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)
            
            # Get top emotions
            top_emotion = sorted_emotions[0]
            second_emotion = sorted_emotions[1]
            
            # Calculate confidence (difference between top and second emotion)
            diff = top_emotion[1] - second_emotion[1]
            
            # Determine if ambiguous (small difference between top emotions)
            is_ambiguous = diff < 15  # If difference is less than 15%, consider ambiguous
            
            # Generate explanation for ambiguous cases
            explanation = None
            if is_ambiguous:
                explanation = f"The facial expression shows a mix of {top_emotion[0]} ({top_emotion[1]:.1f}%) " \
                             f"and {second_emotion[0]} ({second_emotion[1]:.1f}%)."
            
            # Calculate overall confidence
            confidence = min(100, max(0, top_emotion[1]))
            
            return confidence, is_ambiguous, explanation
            
        except Exception as e:
            logger.exception(f"Error detecting emotion confidence: {e}")
            return 50.0, True, "Unable to accurately determine emotion confidence"
    
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
                logger.warning("Image preprocessing failed, using original")
                if len(image.shape) == 3:
                    preprocessed = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                else:
                    preprocessed = image
            
            # Detect faces
            faces = self.detect_faces(preprocessed)
            
            if len(faces) == 0:
                logger.warning("No faces detected in the image")
                return None, None
            
            # Get the largest face (assuming it's the main subject)
            largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
            
            # Extract facial features
            features = self.extract_facial_features(image, largest_face)
            
            # Analyze emotions based on features
            emotions = self.analyze_emotion_from_features(features)
            
            if emotions is None:
                logger.warning("Emotion analysis failed")
                return None, None
                
            # Get confidence and ambiguity
            confidence, is_ambiguous, explanation = self.detect_emotion_confidence(emotions)
            
            # Get dominant emotion
            dominant_emotion = max(emotions.items(), key=lambda x: x[1])[0]
            
            # Calculate weighted sentiment score
            sentiment_score = 0
            for emotion, score in emotions.items():
                sentiment_score += EMOTION_TO_SENTIMENT.get(emotion, 0) * (score / 100)
            
            # Ensure score is between -1 and 1
            sentiment_score = max(-1.0, min(1.0, sentiment_score))
            
            # Create detailed results
            emotion_details = {
                'emotions': emotions,
                'dominant_emotion': dominant_emotion,
                'confidence': confidence,
                'is_ambiguous': is_ambiguous,
                'ambiguity_explanation': explanation,
                'face_rect': [int(x) for x in largest_face]
            }
            
            return sentiment_score, emotion_details
            
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
                emotion_name = EMOTION_DESCRIPTIONS.get(dominant_emotion, dominant_emotion).capitalize()
                label = f"{emotion_name}: {confidence:.1f}%"
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
                emotion_name = EMOTION_DESCRIPTIONS.get(emotion, emotion).capitalize()
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
analyzer = EnhancedFacialExpressionAnalyzer()

def analyze_facial_expression(image):
    """
    Public function to analyze facial expression using enhanced analyzer
    
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