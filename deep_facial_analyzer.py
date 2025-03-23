"""
Advanced Facial Expression Recognition Model
Uses deep learning techniques for accurate emotion detection
"""

import os
import cv2
import logging
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
import io
import base64

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
MODEL_PATH = os.path.join("models", "emotion_model.h5")
FACE_CLASSIFIER_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
IMG_SIZE = 48  # Standard size for emotion recognition models
EMOTION_LABELS = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
EMOTION_TO_SENTIMENT = {
    'angry': -0.7,
    'disgust': -0.6,
    'fear': -0.5,
    'sad': -0.5,
    'surprise': 0.3,
    'neutral': 0.0,
    'happy': 0.8
}

# Ensure model directory exists
os.makedirs("models", exist_ok=True)

class DeepFacialExpressionAnalyzer:
    """
    Advanced facial expression analyzer using deep learning
    """
    
    def __init__(self):
        """Initialize the model"""
        # Load face detection model
        self.face_detector = cv2.CascadeClassifier(FACE_CLASSIFIER_PATH)
        
        # Load or create emotion detection model
        if os.path.exists(MODEL_PATH):
            logger.info(f"Loading pre-trained emotion model from {MODEL_PATH}")
            self.emotion_model = load_model(MODEL_PATH)
        else:
            logger.warning(f"No pre-trained model found at {MODEL_PATH}. Using a pre-trained MobileNetV2 model.")
            self.emotion_model = self._create_mobilenet_model()
        
        # Additional models for facial landmarks (if needed in the future)
        self.has_landmarks_model = False
    
    def _create_mobilenet_model(self):
        """
        Create a MobileNetV2-based model for emotion recognition
        MobileNetV2 is efficient and works well on edge devices
        """
        base_model = MobileNetV2(
            include_top=False, 
            weights='imagenet',
            input_shape=(IMG_SIZE, IMG_SIZE, 3)
        )
        
        # Freeze base model layers
        for layer in base_model.layers:
            layer.trainable = False
            
        # Create the model
        model = Sequential([
            base_model,
            Flatten(),
            Dense(512, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            Dense(256, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            Dense(len(EMOTION_LABELS), activation='softmax')
        ])
        
        # Compile the model
        model.compile(
            optimizer=Adam(learning_rate=0.0001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def preprocess_face(self, face):
        """
        Preprocess a face image for the emotion model
        
        Args:
            face: Cropped face image
            
        Returns:
            preprocessed_face: Normalized and resized face image
        """
        # Resize to expected size
        face_resized = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
        
        # Convert grayscale to RGB if needed (MobileNetV2 expects 3 channels)
        if len(face_resized.shape) == 2:
            face_resized = cv2.cvtColor(face_resized, cv2.COLOR_GRAY2RGB)
        elif face_resized.shape[2] == 1:
            face_resized = cv2.cvtColor(face_resized, cv2.COLOR_GRAY2RGB)
            
        # Preprocess for the model
        face_array = img_to_array(face_resized)
        face_array = preprocess_input(face_array)
        face_array = np.expand_dims(face_array, axis=0)
        
        return face_array
    
    def detect_faces(self, image):
        """
        Detect faces in an image
        
        Args:
            image: OpenCV image (numpy array)
            
        Returns:
            list: List of detected face rectangles (x, y, w, h)
        """
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect faces with multiple scales
        faces = self.face_detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        if len(faces) == 0:
            return []
            
        return faces
    
    def get_facial_landmarks(self, image, face_rect):
        """
        Get facial landmarks for a detected face
        
        Args:
            image: OpenCV image (numpy array)
            face_rect: Face rectangle (x, y, w, h)
            
        Returns:
            dict: Facial landmarks or None if not available
        """
        # Skip if landmark detection is not available
        if not self.has_landmarks_model:
            return None
            
        # This would normally use dlib's landmark detector
        # But we'll skip this for now to avoid the dependency
        return None
    
    def analyze_emotions(self, face_array):
        """
        Analyze emotions in a preprocessed face image
        
        Args:
            face_array: Preprocessed face array
            
        Returns:
            dict: Dictionary of emotion probabilities
        """
        # Predict emotions
        emotion_predictions = self.emotion_model.predict(face_array, verbose=0)[0]
        
        # Map predictions to labels
        emotions = {}
        for i, emotion in enumerate(EMOTION_LABELS):
            emotions[emotion] = float(emotion_predictions[i] * 100)
            
        return emotions
    
    def detect_emotion_confidence(self, emotions):
        """
        Calculate confidence score and determine if emotion is ambiguous
        
        Args:
            emotions: Dictionary of emotion probabilities
            
        Returns:
            tuple: (confidence, is_ambiguous, explanation)
        """
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
            # Detect faces
            faces = self.detect_faces(image)
            
            if len(faces) == 0:
                logger.warning("No faces detected in the image")
                return None, None
            
            # Get the largest face (assuming it's the main subject)
            largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
            x, y, w, h = largest_face
            
            # Extract face ROI
            face_roi = image[y:y+h, x:x+w]
            
            # Preprocess face
            face_array = self.preprocess_face(face_roi)
            
            # Analyze emotions
            emotions = self.analyze_emotions(face_array)
            
            # Get confidence and ambiguity
            confidence, is_ambiguous, explanation = self.detect_emotion_confidence(emotions)
            
            # Calculate weighted sentiment score
            sentiment_score = 0
            for emotion, score in emotions.items():
                sentiment_score += EMOTION_TO_SENTIMENT.get(emotion, 0) * (score / 100)
            
            # Ensure score is between -1 and 1
            sentiment_score = max(-1.0, min(1.0, sentiment_score))
            
            # Create detailed results
            emotion_details = {
                'emotions': emotions,
                'dominant_emotion': max(emotions.items(), key=lambda x: x[1])[0],
                'confidence': confidence,
                'is_ambiguous': is_ambiguous,
                'explanation': explanation,
                'face_rect': [int(x), int(y), int(w), int(h)]
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
            if 'face_rect' in emotion_details:
                x, y, w, h = emotion_details['face_rect']
                
                # Draw face rectangle
                cv2.rectangle(vis_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # Get dominant emotion and confidence
                dominant_emotion = emotion_details['dominant_emotion']
                confidence = emotion_details['confidence']
                
                # Determine text color based on emotion
                color = (0, 255, 0)  # Default: green
                if dominant_emotion in ['angry', 'disgust', 'fear', 'sad']:
                    color = (0, 0, 255)  # Red for negative emotions
                elif dominant_emotion == 'neutral':
                    color = (255, 255, 0)  # Yellow for neutral
                
                # Draw emotion label
                label = f"{dominant_emotion.title()}: {confidence:.1f}%"
                cv2.putText(vis_image, label, (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                # If the emotion is ambiguous, add explanation
                if emotion_details['is_ambiguous'] and emotion_details['explanation']:
                    # Get the second highest emotion
                    emotions = emotion_details['emotions']
                    sorted_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)
                    second_emotion = sorted_emotions[1][0]
                    
                    # Draw mixed emotion indicator
                    mixed_label = f"Mixed: {second_emotion.title()}"
                    cv2.putText(vis_image, mixed_label, (x, y+h+20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Convert to RGB for displaying
            vis_image_rgb = cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB)
            
            # Convert to base64 for web display
            _, buffer = cv2.imencode('.jpg', vis_image_rgb)
            base64_image = base64.b64encode(buffer).decode('utf-8')
            
            return f"data:image/jpeg;base64,{base64_image}"
            
        except Exception as e:
            logger.exception(f"Error creating visualization: {e}")
            return None

# Initialize the analyzer
analyzer = DeepFacialExpressionAnalyzer()

def analyze_facial_expression(image):
    """
    Public function to analyze facial expression using the deep learning model
    
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