import os
import logging
import time
from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
import numpy as np
from werkzeug.utils import secure_filename
import base64
import io
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from sentiment_analyzer import analyze_text_sentiment
from facial_expression_analyzer import analyze_facial_expression
from enhanced_facial_analyzer import analyze_facial_expression as analyze_facial_expression_enhanced
from enhanced_facial_analyzer import visualize_emotion_analysis
from claude_analyzer import analyze_image_with_claude
import cv2

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "default-secret-key")

# Configure upload settings
UPLOAD_FOLDER = 'temp_uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size

# Create temporary folders if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(os.path.join('static', 'temp'), exist_ok=True)

def cleanup_old_files(directory, max_age_hours=24):
    """Remove files older than the specified age from a directory"""
    try:
        current_time = time.time()
        max_age_seconds = max_age_hours * 3600
        
        # Check if directory exists
        if not os.path.exists(directory):
            return
            
        # List all files in the directory
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            
            # Check if it's a file (not a directory)
            if os.path.isfile(file_path):
                # Get file age
                file_age = current_time - os.path.getmtime(file_path)
                
                # Remove if older than max age
                if file_age > max_age_seconds:
                    os.remove(file_path)
                    logger.debug(f"Cleaned up old file: {file_path}")
    except Exception as e:
        logger.error(f"Error cleaning up files: {str(e)}")

# Clean up old temporary files on startup
cleanup_old_files(UPLOAD_FOLDER)
cleanup_old_files(os.path.join('static', 'temp'))

def allowed_file(filename):
    """Check if the file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def combine_sentiments(text_sentiment, facial_sentiment):
    """Combine text and facial sentiment scores"""
    # If we only have one type of sentiment, return that
    if text_sentiment is None:
        return facial_sentiment
    if facial_sentiment is None:
        return text_sentiment
    
    # Simple weighted average (50% text, 50% facial)
    return (text_sentiment + facial_sentiment) / 2

def generate_sentiment_chart(text_sentiment, facial_sentiment, combined_sentiment):
    """Generate a chart comparing sentiment scores"""
    labels = []
    values = []
    
    if text_sentiment is not None:
        labels.append('Text Sentiment')
        values.append(text_sentiment)
    
    if facial_sentiment is not None:
        labels.append('Facial Expression')
        values.append(facial_sentiment)
    
    if combined_sentiment is not None:
        labels.append('Combined Sentiment')
        values.append(combined_sentiment)
    
    if not labels:  # If no data available
        return None
    
    # Create a figure and plot
    plt.figure(figsize=(10, 5))
    bars = plt.bar(labels, values, color=['#6c757d', '#17a2b8', '#28a745'])
    
    # Add title and labels
    plt.title('Sentiment Analysis Results')
    plt.ylabel('Sentiment Score (-1 to 1)')
    plt.ylim(-1, 1)
    
    # Add a horizontal line at y=0 for reference
    plt.axhline(y=0, color='#dee2e6', linestyle='-', alpha=0.3)
    
    # Add value labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        if height < 0:
            va = 'top'
            offset = -0.1
        else:
            va = 'bottom'
            offset = 0.1
        plt.text(bar.get_x() + bar.get_width()/2., height + offset,
                 f'{height:.2f}', ha='center', va=va)
    
    # Save figure to bytes buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', transparent=True, bbox_inches='tight')
    plt.close()
    buf.seek(0)
    
    # Convert to base64 string for embedding in HTML
    img_str = base64.b64encode(buf.getvalue()).decode('utf-8')
    
    return f'data:image/png;base64,{img_str}'

def get_sentiment_description(score):
    """Convert numerical sentiment score to textual description"""
    if score is None:
        return "No data"
    
    if score >= 0.6:
        return "Very Positive"
    elif score >= 0.2:
        return "Positive"
    elif score > -0.2:
        return "Neutral"
    elif score > -0.6:
        return "Negative"
    else:
        return "Very Negative"

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    """Process uploaded file and/or text for sentiment analysis"""
    try:
        # Initialize sentiment values
        text_sentiment = None
        facial_sentiment = None
        facial_emotions = None
        claude_analysis = None
        
        # Check if we have the Anthropic API key
        has_claude_api = os.environ.get('ANTHROPIC_API_KEY') is not None
        
        # Process text input if provided
        text_input = request.form.get('text_input', '').strip()
        if text_input:
            text_sentiment, text_details = analyze_text_sentiment(text_input)
            logger.debug(f"Text sentiment: {text_sentiment}")
        
        # Process image if uploaded
        if 'image_file' in request.files:
            file = request.files['image_file']
            if file.filename != '' and allowed_file(file.filename):
                # Read file once and keep the bytes for multiple uses
                file.seek(0)
                file_content = file.read()
                file_bytes = np.frombuffer(file_content, np.uint8)
                image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                
                # Try Claude's analysis if API key is available
                if has_claude_api:
                    logger.info("Using Claude AI for advanced image analysis")
                    claude_analysis = analyze_image_with_claude(file_content, text_input)
                    
                    # If Claude successfully analyzed the image
                    if claude_analysis:
                        facial_sentiment = claude_analysis['sentiment_score']
                        facial_emotions = claude_analysis['emotions']
                        logger.debug(f"Claude AI sentiment: {facial_sentiment}")
                        session['image_uploaded_no_face'] = False
                        session['used_claude_analysis'] = True
                    else:
                        # Fall back to OpenCV if Claude analysis fails
                        logger.info("Claude analysis failed, falling back to OpenCV")
                        session['used_claude_analysis'] = False
                        facial_sentiment, facial_emotions = analyze_facial_expression(image)
                else:
                    # Use enhanced facial expression analysis
                    logger.info("Using enhanced facial expression analysis")
                    session['used_claude_analysis'] = False
                    
                    # Try enhanced analysis first
                    facial_sentiment, emotion_details = analyze_facial_expression_enhanced(image)
                    
                    if facial_sentiment is not None:
                        # Advanced analysis succeeded
                        facial_emotions = emotion_details['emotions']
                        session['used_advanced_analysis'] = True
                        
                        # Generate visualization and save to file instead of session
                        visualization = visualize_emotion_analysis(image, emotion_details)
                        # Generate unique filename based on timestamp
                        vis_filename = f"emotion_vis_{int(time.time())}.jpg"
                        vis_path = os.path.join('static', 'temp', vis_filename)
                        
                        # Save base64 image to file if it exists
                        if visualization and visualization.startswith('data:image/jpeg;base64,'):
                            # Extract the base64 part
                            img_data = visualization.split(',')[1]
                            with open(vis_path, 'wb') as f:
                                f.write(base64.b64decode(img_data))
                            # Store just the path in session
                            session['emotion_visualization_path'] = vis_path
                        else:
                            session['emotion_visualization_path'] = None
                        
                        # Store additional details for the UI
                        if 'dominant_emotion' in emotion_details:
                            session['dominant_emotion'] = emotion_details['dominant_emotion']
                        if 'confidence' in emotion_details:
                            session['emotion_confidence'] = emotion_details['confidence']
                        if 'is_ambiguous' in emotion_details:
                            session['emotion_ambiguous'] = emotion_details['is_ambiguous']
                        if 'ambiguity_explanation' in emotion_details:
                            session['emotion_explanation'] = emotion_details['ambiguity_explanation']
                    else:
                        # Fall back to basic OpenCV model
                        logger.info("Enhanced analysis failed, falling back to basic OpenCV")
                        session['used_advanced_analysis'] = False
                        facial_sentiment, facial_emotions = analyze_facial_expression(image)
                
                # Check if a face was detected
                if facial_sentiment is None:
                    flash('No human facial expression detected in the uploaded image.', 'warning')
                    # Set a flag to indicate an image was uploaded but no face was detected
                    session['image_uploaded_no_face'] = True
                else:
                    session['image_uploaded_no_face'] = False
            elif file.filename != '':
                flash('Invalid file format. Please upload a PNG or JPEG image.', 'danger')
        
        # Combine sentiments
        combined_sentiment = combine_sentiments(text_sentiment, facial_sentiment)
        
        # Generate chart
        chart_image = generate_sentiment_chart(text_sentiment, facial_sentiment, combined_sentiment)
        
        # Prepare results data
        results = {
            'text_input': text_input if text_input else None,
            'text_sentiment': {
                'score': text_sentiment,
                'description': get_sentiment_description(text_sentiment)
            } if text_sentiment is not None else None,
            'facial_sentiment': {
                'score': facial_sentiment,
                'description': get_sentiment_description(facial_sentiment),
                'emotions': facial_emotions,
                'used_claude': session.get('used_claude_analysis', False),
                'used_advanced_analysis': session.get('used_advanced_analysis', False),
                'dominant_emotion': session.get('dominant_emotion', None),
                'confidence': session.get('emotion_confidence', None),
                'is_ambiguous': session.get('emotion_ambiguous', False),
                'ambiguity_explanation': session.get('emotion_explanation', None),
                'visualization_path': session.get('emotion_visualization_path', None)
            } if facial_sentiment is not None else None,
            'combined_sentiment': {
                'score': combined_sentiment,
                'description': get_sentiment_description(combined_sentiment)
            } if combined_sentiment is not None else None,
            'chart_image': chart_image,
            'image_uploaded_no_face': session.get('image_uploaded_no_face', False),
            'claude_available': has_claude_api
        }
        
        # Check if we have any data to show
        if text_sentiment is None and facial_sentiment is None:
            flash('Please provide either text or an image for analysis.', 'warning')
            return redirect(url_for('index'))
        
        return render_template('results.html', results=results)
    
    except Exception as e:
        logger.exception("Error during analysis")
        flash(f'An error occurred during analysis: {str(e)}', 'danger')
        return redirect(url_for('index'))

@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle file too large error"""
    flash('The uploaded file is too large. Please upload a file smaller than 16MB.', 'danger')
    return redirect(url_for('index'))

@app.errorhandler(404)
def page_not_found(error):
    """Handle 404 errors"""
    return render_template('error.html', error_code=404, error_message="Page not found"), 404

@app.errorhandler(500)
def internal_server_error(error):
    """Handle 500 errors"""
    return render_template('error.html', error_code=500, error_message="Internal server error"), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
