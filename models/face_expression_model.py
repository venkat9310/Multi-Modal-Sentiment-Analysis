import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)

class FaceExpressionAnalyzer:
    """Class for facial expression analysis using OpenCV and pre-trained models"""
    
    def __init__(self):
        """Initialize the face detection and expression recognition models"""
        # Load face detection model
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Emotion mapping
        self.emotions = {
            0: 'Angry',
            1: 'Disgust',
            2: 'Fear',
            3: 'Happy',
            4: 'Sad',
            5: 'Surprise',
            6: 'Neutral'
        }
        
        # Sentiment mapping (map emotions to sentiment scores)
        self.sentiment_mapping = {
            'Angry': -0.8,
            'Disgust': -0.6,
            'Fear': -0.5,
            'Happy': 0.9,
            'Sad': -0.7,
            'Surprise': 0.2,
            'Neutral': 0.0
        }
        
        logger.info("Face expression analyzer initialized")
    
    def detect_faces(self, image):
        """Detect faces in the image"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        return faces, gray
    
    def predict_emotion(self, face_img):
        """
        Simulates emotion prediction using a pre-trained model
        In a real implementation, this would use a proper model like FER or DeepFace
        """
        # This is a simplified emotion prediction simulation
        # In a real implementation, we would load and use a proper pre-trained model
        
        # Simulated prediction based on image properties
        # This is just for demonstration - a real model would analyze facial features
        brightness = np.mean(face_img)
        variance = np.var(face_img)
        
        # Simple heuristic based on image statistics
        # This should be replaced with actual model prediction
        if brightness > 130:
            # Brighter images more likely to be happy
            emotion_idx = 3  # Happy
            confidence = min(0.5 + (brightness - 130) / 100, 0.9)
        elif variance > 2000:
            # High variance might indicate surprise
            emotion_idx = 5  # Surprise
            confidence = min(0.4 + variance / 10000, 0.8)
        elif brightness < 90:
            # Darker images more likely to be sad or angry
            emotion_idx = 4  # Sad
            confidence = min(0.4 + (90 - brightness) / 100, 0.7)
        else:
            # Default to neutral
            emotion_idx = 6  # Neutral
            confidence = 0.6
            
        return emotion_idx, confidence
    
    def analyze(self, image):
        """Analyze facial expressions in the image"""
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
            
            # Resize for consistent analysis
            face_roi = cv2.resize(face_roi, (48, 48))
            
            # Predict emotion
            emotion_idx, confidence = self.predict_emotion(face_roi)
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
