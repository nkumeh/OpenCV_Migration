import cv2
import numpy as np


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
    # initialise filtered condition
    # greyscale_display = False
    # alternate_greyscale = False

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        if not ret:
            print("Frame is empty")
            break

        if current_display == 'greyscale_display':
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        elif current_display == 'alternate_greyscale':  
            # Convert to grayscale first (assuming you haven't already)
            grayscale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Apply thresholding to create a binary image (black and white)
            threshold_value = 127  # Adjust this value as needed (0-255)
            _, frame = cv2.threshold(grayscale_frame, threshold_value, 255, cv2.THRESH_BINARY)

        # Display the resulting frame
        cv2.imshow("Video", frame)

        # Wait for a key press and break if it's 'q'
        key = cv2.waitKey(10) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('g'):
            current_display = 'greyscale_display' if current_display != 'greyscale_display' else 'normal_colour'
        elif key == ord('h'):
            current_display = 'alternate_greyscale' if current_display != 'alternate_greyscale' else 'normal_colour'
        elif key == ord('s'):
            # Save the frame to a file
            frame_filename = f'saved_frame{frame_counter}.jpg'
            cv2.imwrite(frame_filename, frame)
            print(f'saved: {frame_filename} in current directory')
            frame_counter += 1

    # When everything done, release the capture and close any open windows
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
