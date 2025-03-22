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
            
        # Resize if the image is too large
        max_dim = 1024
        height, width = image.shape[:2]
        if max(height, width) > max_dim:
            scale = max_dim / max(height, width)
            image = cv2.resize(image, (int(width * scale), int(height * scale)))
            
        # Apply some basic preprocessing
        # Convert to grayscale if needed for the face detector
        if len(image.shape) == 3 and image.shape[2] == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # Convert back to BGR for consistency
            processed_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        else:
            processed_img = image.copy()
            
        # Apply mild Gaussian blur to reduce noise
        processed_img = cv2.GaussianBlur(processed_img, (5, 5), 0)
        
        # Apply histogram equalization for better feature detection
        if len(processed_img.shape) == 3:
            # For color images, apply CLAHE to the L channel in LAB color space
            lab = cv2.cvtColor(processed_img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            cl = clahe.apply(l)
            limg = cv2.merge((cl, a, b))
            processed_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        else:
            # For grayscale images
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            processed_img = clahe.apply(processed_img)
            
        return processed_img
    
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        return None
