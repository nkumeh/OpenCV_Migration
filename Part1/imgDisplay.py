import cv2

from vidDisplay import apply_negative

def main():
    # Replace 'your_image.jpg' with the path to the image you want to display
    image_path = '/Users/kristineumeh/Desktop/zara_n_pups.png'
    image = cv2.imread(image_path)
    
    curr_dis = 'normal'
    
    if image is not None:
        # Create a window to display the image
        cv2.namedWindow('Image Display', cv2.WINDOW_NORMAL)
       
        # Enter a loop to check for keypresses
        while True:
            modified_frame = image.copy()
            
            if curr_dis == 'negative':  
                modified_frame = apply_negative(image.copy())

             # Display the image
            cv2.imshow('Image Display', modified_frame)
            
            # Wait for a key press for 1 millisecond
            key = cv2.waitKey(1) & 0xFF

            # If 'q' is pressed, quit the program
            if key == ord('q'):
                break
            elif key == ord('n'):
                curr_dis = 'negative' if curr_dis != 'negative' else 'normal'
              
            # Add more functionality based on keypresses here
            # Example: if key == ord('s'): save the image

        # Close all OpenCV windows
        cv2.destroyAllWindows()
    else:
        print(f"Error: Unable to load image '{image_path}'.")

if __name__ == '__main__':
    main()