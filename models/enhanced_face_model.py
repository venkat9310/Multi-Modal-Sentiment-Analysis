"""
Enhanced Facial Expression Recognition Model

This module provides an enhanced facial expression recognition model
with improved accuracy and support for various image formats.
"""

import os
import logging
import cv2
import numpy as np
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedFaceModel:
    """Enhanced facial expression analysis using OpenCV and feature-based detection"""
    
    def __init__(self):
        """Initialize the face detection and expression recognition models"""
        # Load the pre-trained face cascade classifier
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        
        # Initialize emotion categories and mapping
        self.emotions = {0: 'Angry', 1: 'Happy', 2: 'Sad', 3: 'Surprise', 4: 'Fear'}
        self.sentiment_mapping = {
            'Angry': -0.8,    # Strongly negative
            'Happy': 0.8,     # Strongly positive
            'Sad': -0.6,      # Negative
            'Surprise': 0.2,  # Slightly positive
            'Fear': -0.7      # Negative
        }
        
        # Check for a pre-trained custom model
        self.custom_model_path = Path('./models/custom_emotion_model.xml')
        self.using_custom_model = self.custom_model_path.exists()
        
        # For now, we'll disable TensorFlow to avoid compatibility issues
        self.tf_available = False
        self.tf_model = None
        logger.info("Using feature-based model for emotion prediction")
        
        logger.info("Enhanced face expression analyzer initialized")
    
    def detect_faces(self, image):
        """
        Detect faces in the image with improved parameters for different image qualities
        
        Args:
            image: OpenCV image
            
        Returns:
            faces: List of face coordinates (x, y, w, h)
            gray: Grayscale version of the input image
        """
        # Convert to grayscale if image is in color
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
            
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Try to detect faces with default parameters
        faces = self.face_cascade.detectMultiScale(
            blurred,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        # If no faces found, try with more lenient parameters
        if len(faces) == 0:
            logger.info("No faces detected with default parameters, trying with more lenient parameters")
            faces = self.face_cascade.detectMultiScale(
                blurred,
                scaleFactor=1.05,  # More gradual scaling
                minNeighbors=3,    # Require fewer neighbors
                minSize=(20, 20)   # Allow smaller faces
            )
            
        # If still no faces, try with equalized histogram and different parameters
        if len(faces) == 0:
            logger.info("No faces detected with lenient parameters, trying with histogram equalization")
            equalized = cv2.equalizeHist(gray)
            faces = self.face_cascade.detectMultiScale(
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
    
    def predict_emotion(self, face_img):
        """
        Predicts emotion from facial image using the best available model
        
        Args:
            face_img: OpenCV image (grayscale)
            
        Returns:
            emotion_idx: Index of the detected emotion
            confidence: Confidence score (0-1)
        """
        # Try TensorFlow model first if available
        if self.tf_available and self.tf_model is not None:
            try:
                return self._predict_emotion_tf(face_img)
            except Exception as e:
                logger.error(f"TensorFlow prediction error: {str(e)}")
                # Fall back to feature-based approach
        
        # Fallback to feature-based approach
        return self._predict_emotion_features(face_img)
    
    def _predict_emotion_tf(self, face_img):
        """
        Predict emotion using TensorFlow model
        """
        try:
            # Process the face image for TensorFlow
            # Convert to RGB if grayscale
            if len(face_img.shape) == 2:
                face_rgb = cv2.cvtColor(face_img, cv2.COLOR_GRAY2RGB)
            else:
                face_rgb = face_img
                
            # Resize to 48x48 (expected by the model)
            face_resized = cv2.resize(face_rgb, (48, 48))
            
            # Normalize pixel values to [0, 1]
            face_normalized = face_resized / 255.0
            
            # Add batch dimension
            face_batch = np.expand_dims(face_normalized, axis=0)
            
            # Make prediction
            predictions = self.tf_model.predict(face_batch, verbose=0)[0]
            
            # Get the emotion with highest probability
            emotion_idx = np.argmax(predictions)
            confidence = float(predictions[emotion_idx])
            
            logger.info(f"TensorFlow predicted emotion: {self.emotions[emotion_idx]} with confidence {confidence:.2f}")
            return emotion_idx, confidence
            
        except Exception as e:
            logger.error(f"Error in TensorFlow prediction: {str(e)}")
            raise
    
    def _predict_emotion_features(self, face_img):
        """
        Enhanced emotion prediction using facial features
        This analyzes facial characteristics to estimate emotion with improved accuracy
        """
        # Get image properties
        if len(face_img.shape) > 2:
            # Convert to grayscale if needed
            gray_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            height, width = gray_face.shape
        else:
            gray_face = face_img
            height, width = face_img.shape
        
        # Apply histogram equalization to improve contrast
        equalized_face = cv2.equalizeHist(gray_face)
        
        # Extract facial regions more precisely 
        # Upper face (forehead, eyes, brows) - 0-40% of face height
        forehead_region = equalized_face[0:int(height*0.20), :]
        eye_region = equalized_face[int(height*0.20):int(height*0.40), :]
        
        # Middle face (nose, cheeks) - 40-65% of face height
        nose_region = equalized_face[int(height*0.40):int(height*0.65), :]
        
        # Lower face (mouth, chin) - 65-100% of face height
        mouth_region = equalized_face[int(height*0.65):int(height*0.85), :]
        chin_region = equalized_face[int(height*0.85):, :]
        
        # For convenience in calculations:
        upper_face = np.vstack((forehead_region, eye_region))
        lower_face = np.vstack((mouth_region, chin_region))
        
        # Extract facial features using various techniques
        # 1. Edge detection with different kernel sizes for fine and coarse details
        # Fine details (wrinkles, eye corners, lip edges)
        sobelx_fine = cv2.Sobel(equalized_face, cv2.CV_64F, 1, 0, ksize=3)
        sobely_fine = cv2.Sobel(equalized_face, cv2.CV_64F, 0, 1, ksize=3)
        abs_sobelx_fine = np.absolute(sobelx_fine)
        abs_sobely_fine = np.absolute(sobely_fine)
        
        # Coarse details (overall face shape, major contours)
        sobelx_coarse = cv2.Sobel(equalized_face, cv2.CV_64F, 1, 0, ksize=5)
        sobely_coarse = cv2.Sobel(equalized_face, cv2.CV_64F, 0, 1, ksize=5)
        abs_sobelx_coarse = np.absolute(sobelx_coarse)
        abs_sobely_coarse = np.absolute(sobely_coarse)
        
        # 2. Calculate regional statistics
        # Brightness metrics
        mean_brightness = np.mean(equalized_face)
        forehead_brightness = np.mean(forehead_region)
        eye_brightness = np.mean(eye_region)
        mouth_brightness = np.mean(mouth_region)
        chin_brightness = np.mean(chin_region)
        
        # Texture/variance metrics (measure of detail/wrinkles)
        overall_variance = np.var(equalized_face)
        eye_variance = np.var(eye_region)
        mouth_variance = np.var(mouth_region)
        
        # Edge intensity metrics (measure of facial feature definition)
        # Fine details
        eye_edge_y_fine = np.mean(abs_sobely_fine[int(height*0.20):int(height*0.40), :])
        mouth_edge_y_fine = np.mean(abs_sobely_fine[int(height*0.65):int(height*0.85), :])
        
        # Coarse details
        eye_edge_y_coarse = np.mean(abs_sobely_coarse[int(height*0.20):int(height*0.40), :])
        mouth_edge_y_coarse = np.mean(abs_sobely_coarse[int(height*0.65):int(height*0.85), :])
        
        # Contrast metrics (measure of facial expression intensity)
        eye_mouth_contrast = abs(eye_brightness - mouth_brightness)
        forehead_eye_contrast = abs(forehead_brightness - eye_brightness)
        
        # 3. Feature-based classification with enhanced logic:
        
        # HAPPY: High brightness in mouth region, high edge detection in mouth area (smile lines)
        # Smiling typically creates distinctive creases around the mouth and brightens the lower face
        happy_score = (
            mouth_brightness / 255.0 * 0.4 +                # Bright mouth area
            mouth_edge_y_fine / 100.0 * 0.4 +               # Fine details around mouth (smile lines)
            mouth_edge_y_coarse / 100.0 * 0.2 +             # Coarse mouth shape (smile curve)
            (1.0 - eye_mouth_contrast / 50.0) * 0.2         # More uniform brightness between eyes and mouth
        )
        
        # SAD: Lower brightness overall, drooping mouth corners, less variance in mouth
        # Sad expressions typically have less definition around mouth and darker overall appearance
        sad_score = (
            (1.0 - mouth_brightness / 255.0) * 0.3 +        # Darker mouth area
            (1.0 - mouth_variance / 2000.0) * 0.3 +         # Less textural detail around mouth
            (1.0 - mouth_edge_y_fine / 100.0) * 0.3 +       # Fewer fine details around mouth
            eye_mouth_contrast / 50.0 * 0.2                 # Greater contrast between eyes and mouth
        )
        
        # SURPRISE: Raised eyebrows, wide eyes, open mouth
        # Surprise expressions typically have high edge detection in eye and mouth regions
        surprise_score = (
            eye_edge_y_fine / 100.0 * 0.4 +                 # Fine details around eyes (widened eyes)
            forehead_eye_contrast / 50.0 * 0.3 +            # Contrast between forehead and eyes (raised brows)
            mouth_variance / 2000.0 * 0.3 +                 # Variation in mouth area (open mouth)
            eye_variance / 2000.0 * 0.2                     # Variation in eye area (widened eyes)
        )
        
        # ANGRY: Furrowed brows, intense eye region, compressed mouth
        # Angry expressions typically have high edge detection in eye area and low brightness
        angry_score = (
            eye_edge_y_fine / 100.0 * 0.4 +                 # Fine details around eyes (furrowed brows)
            (1.0 - eye_brightness / 255.0) * 0.3 +          # Darker eye area (shadowed by brows)
            eye_variance / 2000.0 * 0.3 +                   # High variation in eye area (intense expression)
            forehead_eye_contrast / 50.0 * 0.2              # Contrast between forehead and eyes
        )
        
        # FEAR: Wide eyes, raised eyebrows, tense mouth, overall higher contrast
        # Fear typically has high variance throughout the face
        fear_score = (
            eye_edge_y_coarse / 100.0 * 0.3 +               # Coarse details around eyes (widened eyes)
            eye_variance / 2000.0 * 0.3 +                   # High variation in eye area
            overall_variance / 2000.0 * 0.3 +               # Overall facial tension
            (chin_brightness / 255.0) * 0.2                 # Often lighter chin area as jaw tenses
        )
        
        # Bias adjustment based on common misclassifications
        # AVIF images often have different color profiles and contrast
        if overall_variance < 500:  # Low variance images often misclassified
            # Reduce likelihood of surprise for low-variance faces
            surprise_score *= 0.7
            # Increase likelihood of neutral emotions
            sad_score *= 1.2
            happy_score *= 1.2
        
        # Normalize scores to [0, 1] range
        max_score = max(happy_score, sad_score, surprise_score, angry_score, fear_score)
        if max_score > 0:
            happy_score /= max_score
            sad_score /= max_score
            surprise_score /= max_score
            angry_score /= max_score
            fear_score /= max_score
        
        # Find the emotion with highest score
        scores = [angry_score, happy_score, sad_score, surprise_score, fear_score]
        emotion_idx = scores.index(max(scores))
        
        # Calculate confidence based on how distinct the top score is
        sorted_scores = sorted(scores, reverse=True)
        if len(sorted_scores) > 1:
            score_diff = sorted_scores[0] - sorted_scores[1]  # Difference between top two emotions
            confidence = max(scores) * 0.6 + score_diff * 0.4 + 0.2  # Blend of top score and differentiation
        else:
            confidence = max(scores) * 0.8 + 0.2
        
        # Clamp confidence to [0, 1]
        confidence = max(0.0, min(1.0, confidence))
        
        # Log prediction details for debugging
        logger.debug(f"Feature-based emotion scores: Happy={happy_score:.2f}, Sad={sad_score:.2f}, " +
                     f"Surprise={surprise_score:.2f}, Angry={angry_score:.2f}, Fear={fear_score:.2f}")
            
        return emotion_idx, confidence
    
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
            
            # Process the first face found (can be extended to handle multiple faces)
            x, y, w, h = faces[0]
            face_roi = gray[y:y+h, x:x+w]
            
            # Get the color face region if available
            face_color = None
            if len(image.shape) == 3:
                face_color = image[y:y+h, x:x+w]
            
            # Use the color face if available for better analysis
            face_for_analysis = face_color if face_color is not None else face_roi
            
            # Resize for consistent analysis if using feature-based approach
            if not self.tf_available:
                face_for_analysis = cv2.resize(face_for_analysis, (48, 48))
            
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
                'confidence': round(confidence, 2),
                'using_tf': self.tf_available and self.tf_model is not None
            }
            
        except Exception as e:
            logger.error(f"Error in facial expression analysis: {str(e)}")
            return {'score': 0, 'label': 'neutral', 'emotions': {}, 'error': str(e)}