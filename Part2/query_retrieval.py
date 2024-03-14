# query_retrieval.py
import cv2
import numpy as np
import csv
from scipy.spatial.distance import euclidean
from feature_extraction import extract_features_with_9x9, extract_features_multi_histogram, extract_features_2d_histogram 


def read_features_from_csv(input_file):
    """Reads features from  a CSV file into a dictionary.

    The  CSV file should contain two columns: the first column should contain the image  names,
    and the second column should contain the feature vectors. The feature vectors should be 
    comma-separated and converted to float32.

    Args :
        input_file: The path to the CSV file .
    Returns:
         A dictionary mapping image names to feature vectors.
    """

    features  = {}

    with open(input_file , mode='r')  as csvfile: 
        csv_reader = csv. reader(csvfile )

        for  row in csv_reader:
            # Get the image  name and feature vector  from the row. 
            image_name = row[0]
            feature_vector = np.array(row[1:], dtype=np.float32)

            # Add  the feature vector  to the dictionary .
            features[image_name] = feature_vector

    return features

def cbir_query(target_image_path, feature_file, N ):
    # extract features with 9x9
    # target_feature = extract_features_with_9x9(target_image_path)
    
    # extract features with histogram
    # target_feature = extract_features_2d_histogram(target_image_path)
    
    # extract features with multi histogram
    target_feature = extract_features_multi_histogram(target_image_path)
    
    database_features = read_features_from_csv(feature_file)
    
    distances = []
    
    for image_name, feature_vector in database_features.items():
        # eucledian distance metric
        # dist = euclidean(target_feature, feature_vector)
        
        # histogram intersection distance metric
        # dist = find_histogram_intersection(target_feature, feature_vector)
        
        # multi-hist intersection dist metric
        dist = find_composite_histogram_intersection(target_feature, feature_vector)

        distances.append((dist, image_name))
    
    distances.sort(key=lambda x: x[0])
    top_matches = [filename for _, filename in distances[:N]]
    
    return top_matches

def find_histogram_intersection(hist1, hist2):
    """
    Computes the histogram intersection between two histograms.
    
    Args:
        hist1 (numpy.ndarray): The first histogram.
        hist2 (numpy.ndarray): The second histogram.
        
    Returns:
        intersection (float): The sum of the minimum values for each bin pair.
    """
    # Calculate the minimum of each bin pair
    min_hist = np.minimum(hist1, hist2)
    intersection = np.sum(min_hist)
    return intersection

def find_composite_histogram_intersection(hists1, hists2, weights=None):
    """
    Computes a composite distance between two sets of histograms using histogram
    intersection and weighted averaging.
    
    Args:
        hists1 (list): The first set of histograms.
        hists2 (list): The second set of histograms.
        weights (list): Optional weights for each histogram comparison.
    
    Returns:
        The composite distance between the two sets of histograms.
    """
    if weights is None:
        weights = [1] * len(hists1)  # Equal weighting if no weights are specified
    
    intersections = [find_histogram_intersection(h1, h2) for h1, h2 in zip(hists1, hists2)]
    weighted_sum = sum(w * i for w, i in zip(weights, intersections))
    composite_distance = weighted_sum / sum(weights)
    
    return composite_distance



if __name__ == "__main__":
    target_image_path = '/Users/kristineumeh/Desktop/projects/OpenCV_Migration/OpenCV_Migration/Part2/similar_photos/pic.0462.jpg'
    feature_file = '/Users/kristineumeh/Desktop/projects/OpenCV_Migration/OpenCV_Migration/similar_image_features_multi_hist.csv'
    top_matches = cbir_query(target_image_path, feature_file, N=5)
    
    for matches in top_matches:
        # Construct the full path to the image
        matched_image_path = f'/Users/kristineumeh/Desktop/projects/OpenCV_Migration/OpenCV_Migration/Part2/similar_photos/{matches}'
        # Load the image
        matched_image = cv2.imread(matched_image_path)
        if matched_image is not None:
            # Display the image
            cv2.imshow('Image Display', matched_image)
            
            key = cv2.waitKey(0) & 0xFF
            # q to quit
            if key == ord('q'):
                break
    cv2.destroyAllWindows()
    
    print("Top matches:", top_matches)
