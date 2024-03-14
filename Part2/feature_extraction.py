# feature_extraction.py
import cv2
import numpy as np
import os
import csv
import glob

import sys
sys.path.append('/Users/kristineumeh/Desktop/projects/OpenCV_Migration/OpenCV_Migration/')
from Part1.vidDisplay import apply_greyscale


def extract_features_with_9x9(image_path):
    # Load image in grayscale
    image = cv2.imread(image_path)
    
    if image is not None:
        greyscale_image = apply_greyscale(image)
    else:
        print(f"Error loading image {image_path}")
    
    # Extract 9x9 center feature
    height, width = greyscale_image.shape
    center_feature = greyscale_image[(height//2)-4:(height//2)+5, (width//2)-4:(width//2)+5].flatten()
    
    # returned flattened features
    return center_feature

def extract_features_2d_histogram(image_path, bins=32, color_space=cv2.COLOR_BGR2HSV):
    """
    Calculates a 2D normalized colour histogram for an image.
    
    Args:
        image_path (str): Path to the input image.
        bins (int): Number of bins per dimension in the histogram.
        color_space (cv2.ColorConversionCodes): Color space conversion code.
        
    Returns:
        histogram (numpy.ndarray): The 2D normalized histogram.
    """
    # Load the image
    image = cv2.imread(image_path)
    # Convert color space
    image = cv2.cvtColor(image, color_space)
    # Calculate the 2D histogram for the first two channels
    hist = cv2.calcHist([image], [0, 1], None, [bins, bins], [0, 180, 0, 256])
    # Normalize the histogram
    hist = cv2.normalize(hist, hist).flatten()
    return hist

def extract_features_multi_histogram(image_path, bins=32, color_space=cv2.COLOR_BGR2HSV):
    """
    Calculates multiple 2D color histograms for an image: one for the whole image and
    another for the center region.
    
    Args:
        image_path (str): Path to the input image.
        bins (int): Number of bins per dimension in the histogram.
        color_space (cv2.ColorConversionCodes): Color space conversion code.
    
    Returns:
        A list of 2D normalized histograms.
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error loading image {image_path}")
        return None  
    image = cv2.cvtColor(image, color_space)
    
    # Whole image histogram
    # hist_whole = extract_features_2d_histogram(image_path, bins=32, color_space=cv2.COLOR_BGR2HSV)
    
    hist_whole = cv2.calcHist([image], [0, 1], None, [bins, bins], [0, 180, 0, 256])
    hist_whole = cv2.normalize(hist_whole, hist_whole).flatten()
    
    # Center region histogram
    h, w = image.shape[:2]
    cX, cY = w // 2, h // 2
    center = image[cY - h // 4:cY + h // 4, cX - w // 4:cX + w // 4]
    
    # hist_center = extract_features_2d_histogram(center, bins=32, color_space=cv2.COLOR_BGR2HSV)
    
    hist_center = cv2.calcHist([center], [0, 1], None, [bins, bins], [0, 180, 0, 256])
    hist_center = cv2.normalize(hist_center, hist_center).flatten()
    
    return [hist_whole, hist_center]


def save_features_to_csv(image_directory, output_file):
    with open(output_file, 'w', newline='') as csvfile:
        
        feature_writer = csv.writer(csvfile)
        
        for image_path in glob.glob(os.path.join(image_directory, '*.jpg')):
            # feature extraction for 9x9 
            # feature = extract_features_with_9x9(image_path)
            
            # feature extraction for hist
            features = extract_features_multi_histogram(image_path)
            if features is not None:  # Make sure features were successfully extracted
               # Flatten the list of NumPy arrays into a single array
                flattened_features = np.concatenate(features).ravel()
                # Directly concatenate the image basename with the flattened features converted to a list
                row = [os.path.basename(image_path)] + flattened_features.tolist()
                feature_writer.writerow(row)
            else:
                print(f"Failed to extract features for {image_path}")

if __name__ == "__main__":
    image_directory = '/Users/kristineumeh/Desktop/projects/OpenCV_Migration/OpenCV_Migration/Part2/similar_photos'
    output_csv = 'similar_image_features_multi_hist.csv'
    save_features_to_csv(image_directory, output_csv)
