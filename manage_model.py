#!/usr/bin/env python3
"""
Facial Expression Model Management Utility

This script provides utilities for managing the facial expression dataset
and models, including image import, preprocessing, and model training.

Usage:
    python manage_model.py [command] [options]

Commands:
    setup               Create dataset directory structure
    import [source_dir] Import images from source directory
    train               Train the model using current dataset
    status              Show dataset and model status
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from utils.dataset_manager import DatasetManager
from models.train_model import ModelTrainer
from models.train_tensorflow_model import TensorFlowModelTrainer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_dataset():
    """Set up the dataset directory structure"""
    manager = DatasetManager()
    success = manager.create_dataset_structure()
    
    if success:
        logger.info("Dataset structure created successfully")
    else:
        logger.error("Failed to create dataset structure")

def import_images(source_dir, emotion=None):
    """Import images from source directory"""
    if not source_dir:
        logger.error("Source directory is required")
        return
    
    manager = DatasetManager()
    imported_count = manager.import_images(source_dir, emotion)
    
    if imported_count > 0:
        logger.info(f"Successfully imported {imported_count} images")
    else:
        logger.error("Failed to import images")

def train_model():
    """Train the model using current dataset"""
    # Ask user which model to train
    print("\nSelect model to train:")
    print("1. Traditional Model (OpenCV-based)")
    print("2. TensorFlow Model (Deep Learning)")
    choice = input("Enter choice (1/2): ").strip()
    
    if choice == "2":
        # Train TensorFlow model
        tf_trainer = TensorFlowModelTrainer()
        success = tf_trainer.train_model()
        
        if success:
            logger.info("TensorFlow model trained successfully")
        else:
            logger.error("Failed to train TensorFlow model")
    else:
        # Train traditional model (default)
        trainer = ModelTrainer()
        success = trainer.train_model()
        
        if success:
            logger.info("Traditional model trained successfully")
        else:
            logger.error("Failed to train traditional model")

def show_status():
    """Show dataset and model status"""
    manager = DatasetManager()
    stats = manager.get_dataset_statistics()
    
    # Display dataset status
    print("\n=== Dataset Status ===")
    print(f"Dataset exists: {stats['exists']}")
    print(f"Total images: {stats['total_images']}")
    
    if stats['categories']:
        print("\nCategories:")
        for category in stats['categories']:
            print(f"  {category['name']}: {category['image_count']} images")
    
    # Check model files
    custom_model_path = Path('./models/custom_emotion_model.xml')
    tf_model_path = Path('./models/facial_expression_model.h5')
    
    print("\n=== Model Status ===")
    print(f"OpenCV XML model: {'Exists' if custom_model_path.exists() else 'Not found'}")
    print(f"TensorFlow H5 model: {'Exists' if tf_model_path.exists() else 'Not found'}")
    
    print("\n=== Usage Guide ===")
    print("1. Create dataset structure:   python manage_model.py setup")
    print("2. Import images:              python manage_model.py import path/to/images [emotion]")
    print("3. Train model:                python manage_model.py train")
    print("4. Check status:               python manage_model.py status")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Facial Expression Model Management")
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Setup command
    setup_parser = subparsers.add_parser('setup', help='Create dataset directory structure')
    
    # Import command
    import_parser = subparsers.add_parser('import', help='Import images from source directory')
    import_parser.add_argument('source_dir', help='Source directory containing images')
    import_parser.add_argument('--emotion', help='Target emotion category')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train the model using current dataset')
    
    # Status command
    status_parser = subparsers.add_parser('status', help='Show dataset and model status')
    
    args = parser.parse_args()
    
    if args.command == 'setup':
        setup_dataset()
    elif args.command == 'import':
        import_images(args.source_dir, args.emotion)
    elif args.command == 'train':
        train_model()
    elif args.command == 'status':
        show_status()
    else:
        # If no command is provided, show status as default
        show_status()

if __name__ == "__main__":
    main()