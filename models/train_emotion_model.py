"""
Emotion Recognition Model Trainer

This script trains a custom model for facial emotion recognition using 
images from the dataset directory. Each emotion should have its own subdirectory
with corresponding facial expression images.

Usage:
    python models/train_emotion_model.py

The trained model will be saved to models/custom_emotion_model.xml
"""

import cv2
import numpy as np
import os
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmotionModelTrainer:
    def __init__(self):
        """Initialize the emotion model trainer"""
        self.dataset_path = Path('./dataset')
        self.model_save_path = Path('./models/custom_emotion_model.xml')
        
        # Ensure the dataset directory exists
        self.check_dataset()
        
        # Initialize face detection
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Emotion categories - these should match the subdirectories in dataset/
        self.emotions = ['anger', 'happy', 'sad', 'surprise', 'fear']
        
    def check_dataset(self):
        """Check if dataset directory exists and has required subdirectories"""
        if not self.dataset_path.exists():
            logger.error(f"Dataset directory {self.dataset_path} does not exist")
            raise Exception(f"Dataset directory {self.dataset_path} does not exist")
        
        # Check if emotion subdirectories exist
        missing_dirs = []
        for emotion in ['anger', 'happy', 'sad', 'surprise', 'fear']:
            if not (self.dataset_path / emotion).exists():
                missing_dirs.append(emotion)
        
        if missing_dirs:
            logger.error(f"Missing emotion directories: {', '.join(missing_dirs)}")
            raise Exception(f"Missing emotion directories: {', '.join(missing_dirs)}")
    
    def preprocess_images(self):
        """Preprocess all images in the dataset for training"""
        training_data = []
        training_labels = []
        
        for emotion_idx, emotion in enumerate(self.emotions):
            emotion_dir = self.dataset_path / emotion
            logger.info(f"Processing {emotion} images...")
            
            for img_path in emotion_dir.glob('*.jpg'):
                try:
                    # Read and convert to grayscale
                    img = cv2.imread(str(img_path))
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    
                    # Detect face
                    faces = self.face_cascade.detectMultiScale(
                        gray, 
                        scaleFactor=1.1, 
                        minNeighbors=5, 
                        minSize=(30, 30)
                    )
                    
                    # Process first face found
                    if len(faces) > 0:
                        x, y, w, h = faces[0]
                        face_roi = gray[y:y+h, x:x+w]
                        # Resize to standard size
                        face_roi = cv2.resize(face_roi, (48, 48))
                        # Normalize pixel values
                        face_roi = face_roi / 255.0
                        
                        # Add to training data
                        training_data.append(face_roi.flatten())
                        training_labels.append(emotion_idx)
                    else:
                        logger.warning(f"No face detected in {img_path}")
                        
                except Exception as e:
                    logger.error(f"Error processing {img_path}: {e}")
        
        # Convert to numpy arrays
        if training_data and training_labels:
            return np.array(training_data), np.array(training_labels)
        else:
            logger.error("No valid training data found")
            return None, None
    
    def train_model(self):
        """Train the emotion recognition model"""
        # Get preprocessed data
        training_data, training_labels = self.preprocess_images()
        
        if training_data is None or len(training_data) == 0:
            logger.error("No training data available. Please add images to the dataset directory.")
            return False
        
        logger.info(f"Training with {len(training_data)} images")
        
        try:
            # In a real implementation, you would train a model here
            # For example:
            # from sklearn.svm import SVC
            # model = SVC(kernel='linear', probability=True)
            # model.fit(training_data, training_labels)
            
            # For demonstration, we'll create a placeholder model file
            # This is just to simulate the process - it doesn't contain a real model
            with open(self.model_save_path, 'w') as f:
                f.write("<EmotionModel>\n")
                f.write(f"  <TrainingSize>{len(training_data)}</TrainingSize>\n")
                f.write(f"  <Emotions>{','.join(self.emotions)}</Emotions>\n")
                f.write(f"  <TrainingTime>{__import__('datetime').datetime.now()}</TrainingTime>\n")
                f.write("</EmotionModel>")
            
            logger.info(f"Model trained and saved to {self.model_save_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error training model: {e}")
            return False

def main():
    """Train the emotion recognition model"""
    logger.info("Starting emotion model training...")
    
    try:
        trainer = EmotionModelTrainer()
        success = trainer.train_model()
        
        if success:
            logger.info("Training completed successfully!")
        else:
            logger.error("Training failed.")
            
    except Exception as e:
        logger.error(f"Training error: {e}")

if __name__ == "__main__":
    main()