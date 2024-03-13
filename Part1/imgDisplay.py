import cv2

def main():
    # Replace 'your_image.jpg' with the path to the image you want to display
    image_path = '/Users/kristineumeh/Desktop/zara_n_pups.png'
    image = cv2.imread(image_path)

    if image is not None:
        # Create a window to display the image
        cv2.namedWindow('Image Display', cv2.WINDOW_NORMAL)

        # Display the image
        cv2.imshow('Image Display', image)

        # Enter a loop to check for keypresses
        while True:
            # Wait for a key press for 1 millisecond
            key = cv2.waitKey(1) & 0xFF

            # If 'q' is pressed, quit the program
            if key == ord('q'):
                break
            # Add more functionality based on keypresses here
            # Example: if key == ord('s'): save the image

        # Close all OpenCV windows
        cv2.destroyAllWindows()
    else:
        print(f"Error: Unable to load image '{image_path}'.")

if __name__ == '__main__':
    main()