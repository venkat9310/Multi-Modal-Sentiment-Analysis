import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)

def process_image(image):
    """
    Process the image for facial expression analysis
    
    Args:
        image: OpenCV image (numpy array)
        
    Returns:
        Processed image ready for facial analysis
    """
    try:
        if image is None:
            logger.error("Image is None")
            return None
            
        # Check if image is valid
        if image.size == 0:
            logger.error("Image has zero size")
            return None
        
        # Ensure image has 3 channels (BGR)
        if len(image.shape) != 3 or image.shape[2] != 3:
            logger.warning("Converting image to 3-channel format")
            if len(image.shape) == 2:  # Grayscale
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            elif image.shape[2] == 4:  # RGBA (with alpha)
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
            
        # Resize if the image is too large
        max_dim = 1024
        height, width = image.shape[:2]
        if max(height, width) > max_dim:
            scale = max_dim / max(height, width)
            image = cv2.resize(image, (int(width * scale), int(height * scale)))
        
        # Make a copy to avoid modifying the original
        processed_img = image.copy()
        
        # Convert to RGB color space if needed (better for facial detection)
        if cv2.imencode('.jpg', processed_img)[1].mean() < 30:  # Very dark image
            logger.info("Applying brightness adjustment to dark image")
            # Increase brightness for dark images
            hsv = cv2.cvtColor(processed_img, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)
            v = cv2.add(v, 50)  # increase brightness
            final_hsv = cv2.merge((h, s, v))
            processed_img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
            
        # Apply mild Gaussian blur to reduce noise
        processed_img = cv2.GaussianBlur(processed_img, (3, 3), 0)
        
        # Apply histogram equalization for better feature detection
        # For color images, apply CLAHE to the L channel in LAB color space
        lab = cv2.cvtColor(processed_img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        processed_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        
        # Log image processing success
        logger.info(f"Successfully processed image of shape {processed_img.shape}")
            
        return processed_img
    
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        return None
