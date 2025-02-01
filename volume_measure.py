import cv2
import torch
import numpy as np
from typing import Dict, List, Tuple
from collections import defaultdict

def load_midas_model():
    """Load MiDaS model for depth estimation"""
    model_type = "DPT_Hybrid"
    midas = torch.hub.load("intel-isl/MiDaS", model_type, trust_repo=True)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    midas.to(device)
    midas.eval()
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
    transform = midas_transforms.dpt_transform if model_type in ["DPT_Large", "DPT_Hybrid"] else midas_transforms.small_transform
    return midas, transform, device

def compute_iou(boxA, boxB):
    """Compute Intersection over Union (IoU) between two bounding boxes"""
    xA, yA, xB, yB = max(boxA[0], boxB[0]), max(boxA[1], boxB[1]), min(boxA[0] + boxA[2], boxB[0] + boxB[2]), min(boxA[1] + boxA[3], boxB[1] + boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea, boxBArea = boxA[2] * boxA[3], boxB[2] * boxB[3]
    return interArea / float(boxAArea + boxBArea - interArea)

def match_wagons(previous_wagons, current_wagons):
    """Match new detections with previously tracked wagons using IoU."""
    matched_wagons = {}

    # Convert list of tuples to dictionary for easier look-up
    current_wagons_dict = {wagon_id: bbox for wagon_id, bbox in current_wagons}

    for prev_id, prev_bbox in previous_wagons.items():
        best_match, best_iou = None, 0.0

        for curr_id, curr_bbox in current_wagons_dict.items():
            iou = compute_iou(prev_bbox, curr_bbox)
            if iou > best_iou and iou > 0.3:  # Threshold for matching
                best_iou, best_match = iou, curr_id

        if best_match is not None:
            matched_wagons[prev_id] = current_wagons_dict.pop(best_match)  # Safe removal

    # Assign new IDs to unmatched wagons
    new_id = max(previous_wagons.keys(), default=0) + 1
    for new_wagon_id, bbox in current_wagons_dict.items():
        matched_wagons[new_id] = bbox
        new_id += 1

    return matched_wagons


def detect_wagons(frame):
    """Detect multiple wagons in frame and return their bounding boxes"""
    gray, blur = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    _, thresh = cv2.threshold(blur, 60, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return sorted([(i + 1, cv2.boundingRect(c)) for i, c in enumerate(contours) if cv2.contourArea(c) > 10000], key=lambda x: x[1][0])

def calculate_material_volume(depth_map, bbox):
    """Calculate volume of material in the wagon using depth information"""
    x, y, w, h = bbox
    roi_depth = depth_map[y:y+h, x:x+w]
    edge_depth = np.mean([roi_depth[:5, :].mean(), roi_depth[-5:, :].mean()])
    return np.sum(np.clip(edge_depth - roi_depth, 0, None)) * 0.001

class WagonTracker:
    def __init__(self):
        self.wagon_volumes, self.measurement_buffer, self.active_wagons, self.last_wagon_positions = {}, defaultdict(list), set(), {}

    def update_wagon(self, wagon_number, volume, position):
        self.active_wagons.add(wagon_number)
        self.measurement_buffer[wagon_number].append(volume)
        self.last_wagon_positions[wagon_number] = position

    def finalize_wagon(self, wagon_number):
        if wagon_number in self.measurement_buffer:
            sorted_measurements = sorted(self.measurement_buffer[wagon_number])
            self.wagon_volumes[wagon_number] = np.mean(sorted_measurements[len(sorted_measurements)//4:])
            del self.measurement_buffer[wagon_number], self.last_wagon_positions[wagon_number]
            self.active_wagons.remove(wagon_number)

def process_moving_wagons(video_path):
    """Process video with multiple moving wagons and track volumes"""
    midas, transform, device, tracker = *load_midas_model(), WagonTracker()
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): raise ValueError(f"Error: Could not open {video_path}")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        detected_wagons = match_wagons(tracker.last_wagon_positions, detect_wagons(frame))
        current_wagon_numbers = set(detected_wagons.keys())

        for wagon_num in list(tracker.active_wagons):
            if wagon_num not in current_wagon_numbers:
                tracker.finalize_wagon(wagon_num)

        for wagon_num, bbox in detected_wagons.items():
            img, input_batch = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), transform(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).to(device)
            with torch.no_grad():
                depth_map = torch.nn.functional.interpolate(midas(input_batch).unsqueeze(1), size=img.shape[:2], mode="bicubic", align_corners=False).squeeze().cpu().numpy()
            tracker.update_wagon(wagon_num, calculate_material_volume(depth_map, bbox), bbox)
            x, y, w, h = bbox
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f"Wagon {wagon_num}", (x, y-30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.putText(frame, f"Vol: {np.mean(tracker.measurement_buffer[wagon_num]):.2f} m3", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        if cv2.waitKey(1) & 0xFF == ord('q'): break

    for wagon_num in list(tracker.active_wagons):
        tracker.finalize_wagon(wagon_num)

    cap.release()
    cv2.destroyAllWindows()
    return tracker.wagon_volumes

def main():
    video_path = "/content/drive/MyDrive/Dataset/o2.mp4"  # Replace with your video path
    wagon_volumes = process_moving_wagons(video_path)
    print("\nWagon Volume Results:")
    for wagon_id, volume in sorted(wagon_volumes.items()):
        print(f"Wagon {wagon_id}: {volume:.2f} cubic meters")

if __name__ == "__main__":
    main()