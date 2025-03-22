import os
import logging
from pathlib import Path
from flask import Flask, render_template, request, jsonify
import base64
import numpy as np
import cv2
from io import BytesIO
# Import the enhanced face model
from models.enhanced_face_model import EnhancedFaceModel
from models.text_sentiment_model import TextSentimentAnalyzer
from utils.image_processing import process_image
from utils.text_processing import preprocess_text

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "default-secret-key")

# Initialize models
face_analyzer = EnhancedFaceModel()
text_analyzer = TextSentimentAnalyzer()

logger = logging.getLogger(__name__)

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    """Analyze image and/or text for sentiment based on what's provided"""
    try:
        # Get the text input
        text_input = request.form.get('text', '')
        
        # Get the image data
        image_data = request.form.get('image', '')
        
        results = {
            'text_sentiment': {'score': 0, 'label': 'neutral'},
            'face_sentiment': {'score': 0, 'label': 'neutral', 'emotions': {}},
            'combined_sentiment': {'score': 0, 'label': 'neutral'},
            'error': None
        }
        
        # Process text if provided
        if text_input:
            preprocessed_text = preprocess_text(text_input)
            text_result = text_analyzer.analyze(preprocessed_text)
            results['text_sentiment'] = text_result
        
        # Process image if provided
        if image_data and image_data.startswith('data:image'):
            try:
                # Log image format for debugging
                image_format = image_data.split(';')[0].split('/')[1] if ';' in image_data and '/' in image_data.split(';')[0] else "unknown"
                logger.info(f"Processing image with format: {image_format}")
                
                # Extract the base64 encoded data
                image_data = image_data.split(",")[1] if "," in image_data else image_data
                image_bytes = base64.b64decode(image_data)
                
                # Convert to OpenCV format
                nparr = np.frombuffer(image_bytes, np.uint8)
                
                # Try different decode flags for different image formats
                img = None
                
                # Special handling for AVIF or other problematic formats
                if image_format.lower() in ['avif', 'webp', 'heic', 'heif']:
                    logger.info(f"Using special handling for {image_format} format")
                    # Try with IMREAD_UNCHANGED first (preserves alpha channel)
                    img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
                    
                    # If failed or resulted in alpha channel, try with COLOR
                    if img is None or (len(img.shape) == 3 and img.shape[2] == 4):
                        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                else:
                    # Standard handling for common formats
                    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                # Check if image was successfully decoded
                if img is None or img.size == 0:
                    logger.error("Failed to decode image")
                    results['error'] = "Unable to process image format. Please try a different image or format (JPG/PNG recommended)."
                    return jsonify(results)
                
                # Log the loaded image shape for debugging
                logger.info(f"Loaded image shape: {img.shape}")
                
                # Process the image
                processed_img = process_image(img)
                
                # Analyze facial expressions
                if processed_img is not None:
                    face_result = face_analyzer.analyze(processed_img)
                    results['face_sentiment'] = face_result
                else:
                    logger.error("Image processing failed")
                    results['error'] = "Image processing failed. Please try a different image."
            except Exception as e:
                logger.error(f"Error processing image: {str(e)}")
                results['error'] = f"Error processing image: {str(e)}"
        
        # Calculate combined sentiment only if both text and image are provided
        has_text = text_input != ''
        has_image = image_data != '' and image_data.startswith('data:image')
        
        if has_text and has_image:
            # Simple weighted average (can be refined based on confidence)
            text_weight = 0.5
            face_weight = 0.5
                
            # Calculate weighted score
            combined_score = (
                (text_weight * results['text_sentiment']['score'] + 
                 face_weight * results['face_sentiment']['score']) / 
                (text_weight + face_weight)
            )
            
            # Determine label based on combined score
            if combined_score >= 0.5:
                combined_label = 'positive'
            elif combined_score <= -0.5:
                combined_label = 'negative'
            else:
                combined_label = 'neutral'
                
            results['combined_sentiment'] = {
                'score': round(combined_score, 2),
                'label': combined_label
            }
        
        return jsonify(results)
    except Exception as e:
        logger.error(f"Analysis error: {str(e)}")
        return jsonify({'error': f"Analysis error: {str(e)}"}), 500

@app.route('/dataset', methods=['GET'])
def dataset_status():
    """Get the status of the dataset and model"""
    try:
        # Check dataset directories
        dataset_path = Path('./dataset')
        dataset_exists = dataset_path.exists()
        
        tf_model_exists = os.path.exists('./models/facial_expression_model.h5')
        xml_model_exists = os.path.exists('./models/custom_emotion_model.xml')
        
        status = {
            'dataset_exists': dataset_exists,
            'categories': [],
            'total_images': 0,
            'tensorflow_model': tf_model_exists,
            'opencv_model': xml_model_exists,
            'model_trained': tf_model_exists or xml_model_exists
        }
        
        if dataset_exists:
            emotion_categories = ['anger', 'happy', 'sad', 'surprise', 'fear']
            categories_status = []
            
            for emotion in emotion_categories:
                emotion_dir = dataset_path / emotion
                image_count = 0
                
                if emotion_dir.exists():
                    image_count = len(list(emotion_dir.glob('*.jpg')))
                    status['total_images'] += image_count
                
                categories_status.append({
                    'name': emotion,
                    'exists': emotion_dir.exists(),
                    'image_count': image_count
                })
            
            status['categories'] = categories_status
        
        return jsonify(status)
    except Exception as e:
        logger.error(f"Dataset status error: {str(e)}")
        return jsonify({'error': f"Dataset status error: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
