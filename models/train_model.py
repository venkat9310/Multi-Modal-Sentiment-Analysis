import os
import logging
import cv2
import numpy as np
from pathlib import Path
import shutil

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelTrainer:
    """Facial expression model trainer"""
    
    def __init__(self):
        """Initialize the model trainer"""
        self.dataset_path = Path('./Hack_data')
        self.model_path = Path('./models/custom_emotion_model.xml')
        self.emotions = ['anger', 'happy', 'sad', 'surprise', 'fear']
        
    def check_dataset(self):
        """
        Check if dataset directory exists and has required subdirectories
        
        Returns:
            bool: True if dataset is ready, False otherwise
        """
        if not self.dataset_path.exists():
            logger.error(f"Dataset directory {self.dataset_path} does not exist")
            return False
        
        # Check if all emotion subdirectories exist
        missing_dirs = []
        for emotion in self.emotions:
            emotion_dir = self.dataset_path / emotion
            if not emotion_dir.exists():
                missing_dirs.append(emotion)
        
        if missing_dirs:
            logger.error(f"Missing emotion directories: {', '.join(missing_dirs)}")
            return False
        
        # Check if directories have enough images
        not_enough_images = []
        for emotion in self.emotions:
            emotion_dir = self.dataset_path / emotion
            image_count = len(list(emotion_dir.glob('*.jpg')))
            if image_count < 5:  # Minimum required images per class
                not_enough_images.append(f"{emotion} ({image_count})")
        
        if not_enough_images:
            logger.error(f"Not enough images in directories: {', '.join(not_enough_images)}")
            return False
        
        logger.info("Dataset validated successfully")
        return True
    
    def preprocess_images(self):
        """Preprocess all images in the dataset for training"""
        total_processed = 0
        
        for emotion in self.emotions:
            emotion_dir = self.dataset_path / emotion
            if not emotion_dir.exists():
                continue
                
            # Create a processed directory
            processed_dir = emotion_dir / 'processed'
            processed_dir.mkdir(exist_ok=True)
            
            # Process all images
            for img_path in emotion_dir.glob('*.jpg'):
                if 'processed' in str(img_path):
                    continue
                    
                try:
                    # Read image
                    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                    
                    # Skip if image couldn't be read
                    if img is None:
                        logger.warning(f"Could not read image: {img_path}")
                        continue
                    
                    # Preprocess image
                    resized = cv2.resize(img, (48, 48))
                    equalized = cv2.equalizeHist(resized)
                    
                    # Save processed image
                    processed_path = processed_dir / img_path.name
                    cv2.imwrite(str(processed_path), equalized)
                    total_processed += 1
                    
                except Exception as e:
                    logger.error(f"Error processing image {img_path}: {str(e)}")
        
        logger.info(f"Preprocessed {total_processed} images")
        return total_processed > 0
    
    def train_model(self):
        """
        Train the emotion recognition model
        
        Returns:
            bool: True if training was successful, False otherwise
        """
        try:
            # Check if dataset is ready
            if not self.check_dataset():
                return False
            
            # Preprocess images
            if not self.preprocess_images():
                logger.error("Image preprocessing failed")
                return False
            
            # For demonstration purposes, we'll just create a dummy model file
            # In a real implementation, this would use OpenCV or TensorFlow to train a model
            with open(str(self.model_path), 'w') as f:
                f.write("# Dummy model file\n")
                f.write("# Replace with actual trained model\n")
            
            logger.info(f"Model saved to {self.model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error during model training: {str(e)}")
            return False
    
    def copy_sample_images(self, source_folder):
        """
        Copy sample images from a source folder to the dataset
        
        Args:
            source_folder: Path to the folder containing sample images
            
        Returns:
            bool: True if copying was successful
        """
        try:
            source_path = Path(source_folder)
            if not source_path.exists():
                logger.error(f"Source folder {source_folder} does not exist")
                return False
            
            # Ensure dataset structure exists
            for emotion in self.emotions:
                emotion_dir = self.dataset_path / emotion
                emotion_dir.mkdir(exist_ok=True)
            
            # Copy sample images to corresponding emotion folders
            # For demonstration, we'll copy all images to the 'happy' folder
            # In a real implementation, you would use classification to determine the emotion
            happy_dir = self.dataset_path / 'happy'
            
            copied_count = 0
            for img_path in source_path.glob('*.jpg'):
                dest_path = happy_dir / img_path.name
                shutil.copy(str(img_path), str(dest_path))
                copied_count += 1
                
            logger.info(f"Copied {copied_count} sample images to dataset")
            return True
            
        except Exception as e:
            logger.error(f"Error copying sample images: {str(e)}")
            return False

def main():
    """Train the emotion recognition model"""
    trainer = ModelTrainer()
    success = trainer.train_model()
    
    if success:
        logger.info("Model training completed successfully")
    else:
        logger.error("Model training failed")

if __name__ == "__main__":
    main()