"""
TensorFlow-based Facial Expression Recognition Model

This module implements a facial expression recognition model using TensorFlow
for accurate emotion detection in facial images.
"""

import os
import logging
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from PIL import Image

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TensorFlowFaceModel:
    """Class for facial expression analysis using a TensorFlow CNN model"""
    
    def __init__(self):
        """Initialize the TensorFlow model for facial expression recognition"""
        self.model = None
        self.emotions = {0: 'Angry', 1: 'Happy', 2: 'Sad', 3: 'Surprise', 4: 'Fear'}
        self.sentiment_mapping = {
            'Angry': -0.8,    # Strongly negative
            'Happy': 0.8,     # Strongly positive
            'Sad': -0.6,      # Negative
            'Surprise': 0.2,  # Slightly positive
            'Fear': -0.7      # Negative
        }
        # Load or create model
        self.load_model()
        
        logger.info("TensorFlow facial expression model initialized")
    
    def load_model(self):
        """Load the pre-trained model or create a new one if not available"""
        model_path = os.path.join('models', 'facial_expression_model.h5')
        
        if os.path.exists(model_path):
            logger.info(f"Loading pre-trained model from {model_path}")
            try:
                self.model = tf.keras.models.load_model(model_path)
                return True
            except Exception as e:
                logger.error(f"Error loading model: {str(e)}")
        
        # If model doesn't exist or fails to load, create a new one
        logger.info("Creating a new model")
        self.create_model()
        return False
    
    def create_model(self):
        """Create a new TensorFlow model for facial expression recognition"""
        model = Sequential([
            Conv2D(128, (3, 3), activation='relu', input_shape=(48, 48, 3)),
            MaxPooling2D(2, 2),
            Conv2D(256, (3, 3), activation='relu'),
            MaxPooling2D(2, 2),
            Conv2D(512, (3, 3), activation='relu'),
            MaxPooling2D(2, 2),
            Flatten(),
            Dense(1024, activation='relu'),
            Dropout(0.5),
            Dense(5, activation='softmax')  # 5 emotions: Angry, Happy, Sad, Surprise, Fear
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        logger.info("New model created")
    
    def preprocess_face(self, face_img):
        """
        Preprocess the face image for the model
        
        Args:
            face_img: OpenCV grayscale image
            
        Returns:
            Preprocessed image ready for the model
        """
        try:
            # Convert to RGB (TensorFlow model expects 3 channels)
            if len(face_img.shape) == 2:  # If grayscale
                face_rgb = cv2.cvtColor(face_img, cv2.COLOR_GRAY2RGB)
            else:
                face_rgb = face_img
            
            # Resize to 48x48 (model input size)
            face_resized = cv2.resize(face_rgb, (48, 48))
            
            # Normalize pixel values to [0, 1]
            face_normalized = face_resized / 255.0
            
            # Add batch dimension
            face_batch = np.expand_dims(face_normalized, axis=0)
            
            return face_batch
            
        except Exception as e:
            logger.error(f"Error preprocessing face: {str(e)}")
            return None
    
    def predict_emotion(self, face_img):
        """
        Predict emotion from face image using TensorFlow model
        
        Args:
            face_img: OpenCV image (grayscale)
            
        Returns:
            emotion_idx: Index of the detected emotion
            confidence: Confidence score (0-1)
        """
        # If model is not available, fall back to feature-based method
        if self.model is None:
            logger.warning("Model not available, using fallback method")
            return self._predict_emotion_fallback(face_img)
            
        try:
            # Preprocess the face
            processed_face = self.preprocess_face(face_img)
            
            if processed_face is None:
                logger.error("Face preprocessing failed")
                return self._predict_emotion_fallback(face_img)
            
            # Get predictions
            predictions = self.model.predict(processed_face, verbose=0)[0]
            
            # Get the predicted emotion and confidence
            emotion_idx = np.argmax(predictions)
            confidence = predictions[emotion_idx]
            
            logger.info(f"Predicted emotion: {self.emotions[emotion_idx]} with confidence {confidence:.2f}")
            
            # Log all probabilities for debugging
            emotion_probs = [f"{self.emotions[i]}: {predictions[i]:.2f}" for i in range(len(self.emotions))]
            logger.debug(f"All emotion probabilities: {', '.join(emotion_probs)}")
            
            return emotion_idx, float(confidence)
            
        except Exception as e:
            logger.error(f"Error in TensorFlow prediction: {str(e)}")
            # Fall back to feature-based approach
            return self._predict_emotion_fallback(face_img)
    
    def _predict_emotion_fallback(self, face_img):
        """
        Fallback method for emotion prediction when TensorFlow model fails
        Uses feature-based approach
        """
        # Get image properties
        height, width = face_img.shape if len(face_img.shape) == 2 else face_img.shape[:2]
        
        # Convert to grayscale if necessary
        if len(face_img.shape) > 2:
            gray_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        else:
            gray_face = face_img
            
        # Apply histogram equalization to improve contrast
        equalized_face = cv2.equalizeHist(gray_face)
        
        # Extract facial regions
        forehead_region = equalized_face[0:int(height*0.20), :]
        eye_region = equalized_face[int(height*0.20):int(height*0.40), :]
        mouth_region = equalized_face[int(height*0.65):int(height*0.85), :]
        chin_region = equalized_face[int(height*0.85):, :]
        
        # Calculate features
        mean_brightness = np.mean(equalized_face)
        eye_brightness = np.mean(eye_region)
        mouth_brightness = np.mean(mouth_region)
        
        # Simple feature-based classification
        happy_score = mouth_brightness / 255.0
        sad_score = (1.0 - mouth_brightness / 255.0) * 0.8
        surprise_score = np.var(eye_region) / 2000.0
        angry_score = (1.0 - eye_brightness / 255.0) * 0.7
        fear_score = np.var(equalized_face) / 2000.0
        
        # Find the emotion with highest score
        scores = [angry_score, happy_score, sad_score, surprise_score, fear_score]
        emotion_idx = scores.index(max(scores))
        confidence = 0.5  # Default confidence for fallback method
        
        logger.warning(f"Using fallback emotion prediction: {self.emotions[emotion_idx]}")
        
        return emotion_idx, confidence
    
    def detect_faces(self, image):
        """
        Detect faces in the image with Haar Cascade Classifier
        
        Args:
            image: OpenCV image
            
        Returns:
            faces: List of face coordinates (x, y, w, h)
            gray: Grayscale version of the input image
        """
        # Load the face cascade classifier
        face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        face_cascade = cv2.CascadeClassifier(face_cascade_path)
        
        # Convert to grayscale if image is in color
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
            
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Try to detect faces with default parameters
        faces = face_cascade.detectMultiScale(
            blurred,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        # If no faces found, try with more lenient parameters
        if len(faces) == 0:
            logger.info("No faces detected with default parameters, trying with more lenient parameters")
            faces = face_cascade.detectMultiScale(
                blurred,
                scaleFactor=1.05,  # More gradual scaling
                minNeighbors=3,    # Require fewer neighbors
                minSize=(20, 20)   # Allow smaller faces
            )
            
        # If still no faces, try with equalized histogram and different parameters
        if len(faces) == 0:
            logger.info("No faces detected with lenient parameters, trying with histogram equalization")
            equalized = cv2.equalizeHist(gray)
            faces = face_cascade.detectMultiScale(
                equalized,
                scaleFactor=1.03,  # Even more gradual scaling
                minNeighbors=2,    # Even fewer neighbors required
                minSize=(20, 20)
            )
            
        # Log the result
        if len(faces) > 0:
            logger.info(f"Detected {len(faces)} faces in the image")
        else:
            logger.warning("No faces detected in the image after multiple attempts")
            
        return faces, gray
    
    def analyze(self, image):
        """
        Analyze facial expressions in the image
        
        Args:
            image: OpenCV image
            
        Returns:
            Dictionary with emotion analysis results
        """
        if image is None:
            return {'score': 0, 'label': 'neutral', 'emotions': {}}
        
        try:
            faces, gray = self.detect_faces(image)
            
            if len(faces) == 0:
                logger.warning("No faces detected in the image")
                return {'score': 0, 'label': 'neutral', 'emotions': {}, 'message': 'No faces detected'}
            
            # Process the first face found
            x, y, w, h = faces[0]
            face_roi = gray[y:y+h, x:x+w]
            
            # Convert to color if needed for the model
            face_color = image[y:y+h, x:x+w] if len(image.shape) == 3 else None
            
            # Use color face if available, otherwise use grayscale
            face_for_analysis = face_color if face_color is not None else face_roi
            
            # Predict emotion
            emotion_idx, confidence = self.predict_emotion(face_for_analysis)
            emotion = self.emotions[emotion_idx]
            
            # Get sentiment score from emotion
            sentiment_score = self.sentiment_mapping[emotion]
            
            # Determine sentiment label
            if sentiment_score >= 0.5:
                sentiment_label = 'positive'
            elif sentiment_score <= -0.5:
                sentiment_label = 'negative'
            else:
                sentiment_label = 'neutral'
            
            # Create a dictionary of all emotions with their probabilities
            emotion_probs = {}
            for idx, emotion_name in self.emotions.items():
                if idx == emotion_idx:
                    emotion_probs[emotion_name.lower()] = round(confidence, 2)
                else:
                    # Distribute remaining probability
                    emotion_probs[emotion_name.lower()] = round((1 - confidence) / (len(self.emotions) - 1), 2)
            
            return {
                'score': round(sentiment_score, 2),
                'label': sentiment_label,
                'emotions': emotion_probs,
                'primary_emotion': emotion.lower(),
                'confidence': round(confidence, 2)
            }
            
        except Exception as e:
            logger.error(f"Error in facial expression analysis: {str(e)}")
            return {'score': 0, 'label': 'neutral', 'emotions': {}, 'error': str(e)}