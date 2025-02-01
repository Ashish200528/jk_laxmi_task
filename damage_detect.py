import cv2 as cv
import numpy as np

def rescaleFrame(frame, scale=0.6):
    """Rescales a frame (image or video) by a given scale factor."""
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width, height)
    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

def get_roi(frame):
    """Extracts a polygon-shaped region of interest (ROI) from the given frame."""
    # Define the polygon vertices
    pts = np.array([[845, 14], [530, 1780], [2306, 1750], [1841, 7]], np.int32)
    pts = pts.reshape((-1, 1, 2))

    # Create a mask for the ROI
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv.fillPoly(mask, [pts], 255)

    # Apply the mask to the frame
    roi = cv.bitwise_and(frame, frame, mask=mask)

    # Get the bounding rectangle of the ROI
    x, y, w, h = cv.boundingRect(pts)
    cropped_roi = roi[y:y+h, x:x+w]

    return cropped_roi, (x, y), pts

# Load the video
capture = cv.VideoCapture(r"D:\programs\Python\hackathon_nirma\Usecase_Dataset\Usecase_Dataset\IN\2.mp4")

# Check if the video is opened successfully
if not capture.isOpened():
    print('Error: Unable to open the video')
    exit(0)

while True:
    # Read a frame from the video
    isTrue, frame = capture.read()
    if not isTrue:
        print("Error: Unable to read the frame.")
        break

    # Resize the frame
    frame_resized = rescaleFrame(frame)

    # Extract the Region of Interest (ROI)
    roi_frame, (x_offset, y_offset), pts = get_roi(frame)

    # Convert to grayscale and detect edges
    gray = cv.cvtColor(roi_frame, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray, (5, 5), 0)
    canny = cv.Canny(blur, 180, 230)

    # Find and draw contours on the ROI frame
    contours, hierarchy = cv.findContours(canny, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    cv.drawContours(roi_frame, contours, -1, (0, 0, 255), 1)

    # Display the processed ROI frame
    cv.imshow('Contours on ROI', roi_frame)

    # Exit on pressing 'd'
    if cv.waitKey(20) & 0xFF == ord('d'):
        break

# Release resources
capture.release()
cv.destroyAllWindows()
