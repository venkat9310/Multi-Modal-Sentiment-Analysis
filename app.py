import os
import logging
from flask import Flask, render_template, request, jsonify
import base64
import numpy as np
import cv2
from io import BytesIO
from models.face_expression_model import FaceExpressionAnalyzer
from models.text_sentiment_model import TextSentimentAnalyzer
from utils.image_processing import process_image
from utils.text_processing import preprocess_text

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "default-secret-key")

# Initialize models
face_analyzer = FaceExpressionAnalyzer()
text_analyzer = TextSentimentAnalyzer()

logger = logging.getLogger(__name__)

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    """Analyze both image and text for sentiment"""
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
                # Extract the base64 encoded data
                image_data = image_data.split(",")[1]
                image_bytes = base64.b64decode(image_data)
                
                # Convert to OpenCV format
                nparr = np.frombuffer(image_bytes, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                # Process the image
                processed_img = process_image(img)
                
                # Analyze facial expressions
                if processed_img is not None:
                    face_result = face_analyzer.analyze(processed_img)
                    results['face_sentiment'] = face_result
            except Exception as e:
                logger.error(f"Error processing image: {str(e)}")
                results['error'] = f"Error processing image: {str(e)}"
        
        # Calculate combined sentiment
        if results['text_sentiment']['score'] != 0 or results['face_sentiment']['score'] != 0:
            # Simple weighted average (can be refined based on confidence)
            text_weight = 0.5 if text_input else 0
            face_weight = 0.5 if image_data else 0
            
            # Adjust weights if only one input is provided
            if text_input and not image_data.startswith('data:image'):
                text_weight = 1.0
                face_weight = 0.0
            elif image_data.startswith('data:image') and not text_input:
                text_weight = 0.0
                face_weight = 1.0
                
            # Calculate weighted score
            if text_weight + face_weight > 0:
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

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
