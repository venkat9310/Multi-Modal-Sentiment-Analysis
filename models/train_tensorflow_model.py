"""
TensorFlow Emotion Recognition Model Trainer

This script trains a TensorFlow-based model for facial emotion recognition
using images from the dataset directory. Each emotion should have its own
subdirectory with corresponding facial expression images.

Usage:
    python models/train_tensorflow_model.py

The trained model will be saved to models/facial_expression_model.h5
"""

import os
import logging
import cv2
import numpy as np
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TensorFlowModelTrainer:
    """TensorFlow emotion recognition model trainer"""
    
    def __init__(self):
        """Initialize the emotion model trainer"""
        self.dataset_path = Path('./Hack_data')
        self.model_path = Path('./models/facial_expression_model.h5')
        self.emotions = ['anger', 'happy', 'sad', 'surprise', 'fear']
        self.img_size = (48, 48)
        self.batch_size = 64
        self.epochs = 30
        self.validation_split = 0.2
        self.model = None
        
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
            if image_count < 10:  # Minimum required images per class
                not_enough_images.append(f"{emotion} ({image_count})")
        
        if not_enough_images:
            logger.error(f"Not enough images in directories: {', '.join(not_enough_images)}")
            return False
        
        logger.info("Dataset validated successfully")
        return True
    
    def create_data_generators(self):
        """
        Create training and validation data generators
        
        Returns:
            tuple: (train_generator, val_generator)
        """
        # Data augmentation configuration
        datagen = ImageDataGenerator(
            rescale=1./255,
            validation_split=self.validation_split,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        
        # Load training data
        train_generator = datagen.flow_from_directory(
            self.dataset_path,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='training',
            color_mode='rgb'
        )
        
        # Load validation data
        val_generator = datagen.flow_from_directory(
            self.dataset_path,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='validation',
            color_mode='rgb'
        )
        
        return train_generator, val_generator
    
    def build_model(self, num_classes):
        """
        Build the TensorFlow model architecture
        
        Args:
            num_classes: Number of emotion classes
            
        Returns:
            TensorFlow model
        """
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
            Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.0001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
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
            
            # Create data generators
            train_generator, val_generator = self.create_data_generators()
            
            # Build model
            num_classes = len(self.emotions)
            self.model = self.build_model(num_classes)
            
            # Define early stopping callback
            early_stopping = EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            )
            
            # Train model
            logger.info(f"Starting model training for {self.epochs} epochs...")
            history = self.model.fit(
                train_generator,
                validation_data=val_generator,
                epochs=self.epochs,
                callbacks=[early_stopping]
            )
            
            # Save model
            self.model.save(str(self.model_path))
            logger.info(f"Model saved to {self.model_path}")
            
            # Report final metrics
            final_acc = history.history['accuracy'][-1]
            final_val_acc = history.history['val_accuracy'][-1]
            logger.info(f"Training accuracy: {final_acc:.4f}")
            logger.info(f"Validation accuracy: {final_val_acc:.4f}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error during model training: {str(e)}")
            return False

def main():
    """Train the emotion recognition model"""
    trainer = TensorFlowModelTrainer()
    success = trainer.train_model()
    
    if success:
        logger.info("Model training completed successfully")
    else:
        logger.error("Model training failed")

if __name__ == "__main__":
    main()