import cv2
import numpy as np


def apply_greyscale(frame):
    """Converts the frame to greyscale using OpenCV's built-in function."""
    
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

def apply_monochrome(frame, threshold_value=127):
    """Converts the frame to black and white using thresholding."""
    
    grayscale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, monochrome_frame = cv2.threshold(grayscale_frame, threshold_value, 255, cv2.THRESH_BINARY)
    return monochrome_frame

def apply_blur(frame):
    """Applies a blur using various blur filters.
    Args:
        src (cv2.Mat): The input color image (BGR format).
        
    Returns:
        dst (cv2.Mat): The output color image (BGR format).
    """
    
    # gaussian blur
    # blurred_image = cv2.medianBlur(src, 5)
    
    # median blur
    # blurred_image = cv2.medianBlur(src, 5)
    
    # bilateral blur
    # blurred_image = cv2.bilateralFilter(src, 9, 75, 75)
    
    # box blur
    blurred_image = cv2.blur(frame, (5, 5))
    
    return blurred_image
    
def apply_sobel_x(frame):
    kernel = np.array([[-1, 0, 1],
                       [-2, 0, 2],
                       [-1, 0, 1]], dtype=np.float32)
    sobel_x = cv2.filter2D(frame, -1, kernel)
    return sobel_x
    # return cv2.Sobel(frame, cv2.CV_64F, 1, 0, ksize=3)


def apply_sobel_y(frame):
    kernel = np.array([[-1, -2, -1],
                       [ 0,  0,  0],
                       [ 1,  2,  1]], dtype=np.float32)
    sobel_y = cv2.filter2D(frame, -1, kernel)
    return sobel_y
    # image = apply_greyscale(frame)
    # return cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

def apply_magnitude(frame):
    # frame = apply_greyscale(frame)  # Read as greyscale
    return cv2.magnitude(apply_sobel_x(frame).astype(np.float32), apply_sobel_y(frame).astype(np.float32))
    # return cv2.sqrt(cv2.addWeighted(cv2.pow(apply_sobel_x(frame).astype(np.float32), 2), 1, cv2.pow(apply_sobel_y(frame).astype(np.float32), 2), 1, 0))

def apply_blur_quantize(frame, levels):
    # blur image
    blurred_image = apply_blur(frame)
    
    # quantize
    bucket_size = 255 / levels
    
    # Quantize each color channel
    quantized_image = np.floor(blurred_image / bucket_size) * bucket_size + bucket_size / 2
    quantized_image = np.clip(quantized_image, 0, 255)  # Ensure values are within [0, 255]
    
    # Ensure the data type is uint8 as expected by OpenCV
    return np.uint8(quantized_image)

def apply_cartoon(frame, levels, mag_threshold):
    quantized_image = apply_blur_quantize(frame, levels)
    grey_frame = apply_greyscale(frame)

    gradient_mag = apply_magnitude(grey_frame)
    edges = gradient_mag > mag_threshold
    
    cartooned = quantized_image.copy()
    cartooned[edges] = (0,0,0)
    
    return cartooned

def apply_negative(frame):
    return 255 - frame

def apply_sparkles(image, edge_threshold, sparkle_intensity):
      # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Use the Canny edge detector to find edges in the image
    edges = cv2.Canny(gray_image, threshold1=50, threshold2=edge_threshold)
    
    # Identify points where to add sparkles (strong edges)
    # Here, we simply use the edge locations detected by Canny
    y_coords, x_coords = np.where(edges > 0)
    
    # Add sparkles to the image
    sparkle_image = image.copy()
    for x, y in zip(x_coords, y_coords):
        sparkle_image[y, x] = [sparkle_intensity] * 3  # Set the pixel to bright white
    
    return sparkle_image
    

def main():
    # Open the video device
    cap = cv2.VideoCapture(0)  # webcam
    # cap = cv2.VideoCapture(1) # phonecam

    if not cap.isOpened():
        print("Unable to open video device")
        return

    # Get some properties of the video
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Expected size: {frame_width} {frame_height}")

    # Create a window for display
    cv2.namedWindow("Video", cv2.WINDOW_AUTOSIZE)

    # initialise frame counter
    frame_counter = 1

    # initialise base frame as colour
    current_display = 'normal_colour'


    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        if not ret:
            print("Frame is empty")
            break
        
        modified_frame = frame.copy()

        if current_display == 'greyscale_display':
            modified_frame = apply_greyscale(frame.copy())
        elif current_display == 'alternate_greyscale':  
            modified_frame = apply_monochrome(frame.copy())
        elif current_display == 'blur':  
            modified_frame = apply_blur(frame.copy())
        elif current_display == 'x_sobel':  
            modified_frame = apply_sobel_x(frame.copy())
        elif current_display == 'y_sobel':  
            modified_frame = apply_sobel_y(frame.copy())
        elif current_display == 'magnitude':  
            modified_frame = apply_magnitude(frame.copy())
            modified_frame = cv2.normalize(modified_frame, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
            modified_frame = np.uint8(modified_frame)
        elif current_display == 'blur_quantize':  
            modified_frame = apply_blur_quantize(frame.copy(), 10)
        elif current_display == 'cartoon':  
            modified_frame = apply_cartoon(frame.copy(), 10, 15)  
        elif current_display == 'negative':  
            modified_frame = apply_negative(frame.copy())     
        elif current_display == 'sparkle':  
            modified_frame = apply_sparkles(frame.copy(), 100, 255)  
                        
        # Display the resulting frame
        cv2.imshow("Video", modified_frame)

        # Wait for a key press and break if it's 'q'
        key = cv2.waitKey(10) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('g'):
            current_display = 'greyscale_display' if current_display != 'greyscale_display' else 'normal_colour'
        elif key == ord('h'):
            current_display = 'alternate_greyscale' if current_display != 'alternate_greyscale' else 'normal_colour'
        elif key == ord('b'):
            current_display = 'blur' if current_display != 'blur' else 'normal_colour'
        elif key == ord('x'):
            current_display = 'x_sobel' if current_display != 'x_sobel' else 'normal_colour'
        elif key == ord('y'):
            current_display = 'y_sobel' if current_display != 'y_sobel' else 'normal_colour'
        elif key == ord('m'):
            current_display = 'magnitude' if current_display != 'magnitude' else 'normal_colour'
        elif key == ord('l'):
            current_display = 'blur_quantize' if current_display != 'blur_quantize' else 'normal_colour'
        elif key == ord('c'):
            current_display = 'cartoon' if current_display != 'cartoon' else 'normal_colour'
        elif key == ord('n'):
            current_display = 'negative' if current_display != 'negative' else 'normal_colour'
        elif key == ord('p'):
            current_display = 'sparkle' if current_display != 'sparkle' else 'normal_colour'
        elif key == ord('s'):
            # Save the frame to a file
            frame_filename = f'saved_frame{frame_counter}.jpg'
            cv2.imwrite(frame_filename, modified_frame)
            print(f'saved: {frame_filename} in current directory')
            frame_counter += 1

    # Release the capture and close any open windows
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
