import os
import logging
import cv2
import numpy as np
from pathlib import Path
import shutil

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatasetManager:
    """Facial expression dataset manager"""
    
    def __init__(self):
        """Initialize the dataset manager"""
        self.dataset_path = Path('./Hack_data')
        self.emotions = ['anger', 'happy', 'sad', 'surprise', 'fear']
        
    def create_dataset_structure(self):
        """Create the dataset directory structure"""
        try:
            # Create main dataset directory
            self.dataset_path.mkdir(exist_ok=True)
            
            # Create subdirectories for each emotion
            for emotion in self.emotions:
                emotion_dir = self.dataset_path / emotion
                emotion_dir.mkdir(exist_ok=True)
                logger.info(f"Created directory for {emotion} emotion")
            
            return True
        except Exception as e:
            logger.error(f"Error creating dataset structure: {str(e)}")
            return False
    
    def import_images(self, source_dir, target_emotion=None):
        """
        Import images from a source directory to the dataset
        
        Args:
            source_dir: Path to the source directory
            target_emotion: Specific emotion to import images to (optional)
            
        Returns:
            int: Number of imported images
        """
        try:
            source_path = Path(source_dir)
            if not source_path.exists():
                logger.error(f"Source directory does not exist: {source_dir}")
                return 0
            
            # Create dataset structure if it doesn't exist
            self.create_dataset_structure()
            
            # If target emotion is specified, import to that emotion
            if target_emotion and target_emotion in self.emotions:
                target_dirs = [self.dataset_path / target_emotion]
            else:
                # Otherwise, try to detect emotion from directory name
                target_dirs = []
                for emotion in self.emotions:
                    if emotion in str(source_path).lower():
                        target_dirs.append(self.dataset_path / emotion)
                        break
                
                # If no emotion detected, use 'happy' as default
                if not target_dirs:
                    target_dirs = [self.dataset_path / 'happy']
            
            # Import images
            imported_count = 0
            for target_dir in target_dirs:
                # Get list of image files
                image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
                image_files = []
                for ext in image_extensions:
                    image_files.extend(list(source_path.glob(f'*{ext}')))
                    image_files.extend(list(source_path.glob(f'*{ext.upper()}')))
                
                # Copy each image
                for img_path in image_files:
                    target_path = target_dir / img_path.name
                    shutil.copy(str(img_path), str(target_path))
                    imported_count += 1
                    
                    # Verify the image can be read
                    try:
                        img = cv2.imread(str(target_path))
                        if img is None:
                            logger.warning(f"Copied file but could not read as image: {img_path.name}")
                            os.remove(str(target_path))
                            imported_count -= 1
                    except Exception as e:
                        logger.warning(f"Failed to verify image {img_path.name}: {str(e)}")
                        os.remove(str(target_path))
                        imported_count -= 1
                
            logger.info(f"Imported {imported_count} images")
            return imported_count
            
        except Exception as e:
            logger.error(f"Error importing images: {str(e)}")
            return 0
    
    def preprocess_dataset(self):
        """
        Preprocess all images in the dataset
        
        Returns:
            bool: True if preprocessing was successful
        """
        try:
            # Check if dataset exists
            if not self.dataset_path.exists():
                logger.error("Dataset directory does not exist")
                return False
            
            # Process each emotion directory
            total_processed = 0
            for emotion in self.emotions:
                emotion_dir = self.dataset_path / emotion
                if not emotion_dir.exists():
                    continue
                
                # Create processed directory
                processed_dir = emotion_dir / 'processed'
                processed_dir.mkdir(exist_ok=True)
                
                # Process all images
                for img_path in emotion_dir.glob('*.jpg'):
                    if 'processed' in str(img_path):
                        continue
                    
                    try:
                        # Read image
                        img = cv2.imread(str(img_path))
                        
                        # Skip if image couldn't be read
                        if img is None:
                            logger.warning(f"Could not read image: {img_path}")
                            continue
                        
                        # Convert to grayscale
                        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        
                        # Resize to 48x48
                        resized = cv2.resize(gray, (48, 48))
                        
                        # Apply histogram equalization
                        equalized = cv2.equalizeHist(resized)
                        
                        # Save processed image
                        processed_path = processed_dir / img_path.name
                        cv2.imwrite(str(processed_path), equalized)
                        total_processed += 1
                        
                    except Exception as e:
                        logger.error(f"Error processing image {img_path}: {str(e)}")
            
            logger.info(f"Preprocessed {total_processed} images")
            return total_processed > 0
            
        except Exception as e:
            logger.error(f"Error preprocessing dataset: {str(e)}")
            return False
    
    def get_dataset_statistics(self):
        """
        Get statistics about the dataset
        
        Returns:
            dict: Dataset statistics
        """
        try:
            # Check if dataset exists
            if not self.dataset_path.exists():
                return {
                    'exists': False,
                    'total_images': 0,
                    'categories': []
                }
            
            # Get statistics for each emotion
            categories = []
            total_images = 0
            
            for emotion in self.emotions:
                emotion_dir = self.dataset_path / emotion
                
                # Count images
                image_count = 0
                if emotion_dir.exists():
                    image_count = len(list(emotion_dir.glob('*.jpg')))
                    total_images += image_count
                
                categories.append({
                    'name': emotion,
                    'exists': emotion_dir.exists(),
                    'image_count': image_count
                })
            
            return {
                'exists': True,
                'total_images': total_images,
                'categories': categories
            }
            
        except Exception as e:
            logger.error(f"Error getting dataset statistics: {str(e)}")
            return {
                'exists': False,
                'total_images': 0,
                'categories': [],
                'error': str(e)
            }

# Example usage
if __name__ == "__main__":
    manager = DatasetManager()
    manager.create_dataset_structure()
    stats = manager.get_dataset_statistics()
    print(f"Dataset statistics: {stats}")