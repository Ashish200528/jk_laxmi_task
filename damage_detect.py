import cv2 as cv
import numpy as np

def rescaleFrame(frame, scale=0.6):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width, height)
    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

def get_roi(frame):
    pts = np.array([[845, 14], [530, 1780], [2306, 1750], [1841, 7]], np.int32)
    pts = pts.reshape((-1, 1, 2))
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv.fillPoly(mask, [pts], 255)
    roi = cv.bitwise_and(frame, frame, mask=mask)
    x, y, w, h = cv.boundingRect(pts)
    cropped_roi = roi[y:y+h, x:x+w]
    return cropped_roi, (x, y), pts

def classify_damage(area):
    if area < 500:
        return "small"
    elif 500 <= area < 1500:
        return "medium"
    else:
        return "large"

def process_frame(frame, damage_summary):
    roi_frame, (x_offset, y_offset), pts = get_roi(frame)
    gray = cv.cvtColor(roi_frame, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray, (5, 5), 0)
    canny = cv.Canny(blur, 180, 230)
    contours, _ = cv.findContours(canny, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        area = cv.contourArea(contour)
        if area > 100:
            x, y, w, h = cv.boundingRect(contour)
            damage_severity = classify_damage(area)
            wagon_id = f'wagon{x // 500 + 1}'
            if wagon_id not in damage_summary:
                damage_summary[wagon_id] = []
            damage_summary[wagon_id].append(damage_severity)
            
            cv.drawContours(roi_frame, [contour], -1, (0, 255, 0), 2)
            cv.putText(roi_frame, damage_severity, (x, y-5), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return roi_frame

def get_highest_damage(damage_list):
    if "large" in damage_list:
        return "large"
    elif "medium" in damage_list:
        return "medium"
    else:
        return "small"

capture = cv.VideoCapture('D:/Education Related/JK Lakshmi/Usecase_Dataset/Usecase_Dataset/IN/2.mp4')
if not capture.isOpened():
    print('Error: Unable to open the video')
    exit(0)

damage_summary = {}

while True:
    isTrue, frame = capture.read()
    if not isTrue:
        print("Error: Unable to read the frame.")
        break
    
    processed_frame = process_frame(frame, damage_summary)
    cv.imshow('Processed Frame', processed_frame)
    
    if cv.waitKey(1) & 0xFF == ord('d'):
        break

capture.release()
cv.destroyAllWindows()

for wagon, damages in damage_summary.items():
    print(f"{wagon}: {get_highest_damage(damages)}")
