# Import necessary libraries
import os
import numpy as np
import cv2
import open3d as o3d
from threading import Thread, Lock
import time
from ultralytics import YOLO
import torch

# ObjectDetector class for performing object detection using YOLO
class ObjectDetector:
    def __init__(self):
        # Load the YOLOv8 model (pre-trained weights)
        self.model = YOLO('yolov8n.pt')
        # Get class labels used by YOLO
        self.classes = self.model.names
        
    def detect(self, frame):
        # Perform object detection on the provided frame
        results = self.model(frame, stream=True)
        detections = []
        
        # Parse detection results (bounding boxes, confidences, and class ids)
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]  # Extract coordinates of the bounding box
                confidence = box.conf[0]       # Confidence score of detection
                class_id = box.cls[0]          # Class id of the detected object
                detections.append({
                    'bbox': (int(x1), int(y1), int(x2), int(y2)),  # Bounding box coordinates
                    'confidence': float(confidence),                # Confidence score
                    'class': self.classes[int(class_id)]            # Object class name
                })
        return detections

# DepthEstimator class for estimating depth from the image using edge detection techniques
class DepthEstimator:
    def __init__(self):
        self.last_depth = None
        
    def estimate(self, frame):
        # Convert the input frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Sobel operator to find edges in both x and y directions
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        # Calculate the magnitude of gradients (edge strengths)
        magnitude = np.sqrt(np.square(sobelx) + np.square(sobely))
        
        # Apply Laplacian to further estimate image sharpness
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        
        # Combine the edge magnitude and Laplacian to create the depth estimate
        depth = magnitude + np.abs(laplacian)
        
        # Normalize the depth map to fit into a specific range (0 to 65535)
        depth = cv2.normalize(depth, None, 0, 65535, cv2.NORM_MINMAX)
        depth = np.uint16(depth)  # Convert to 16-bit unsigned integer
        return depth

# ObjectRenderer class for rendering the closest detected object into a 3D point cloud
class ObjectRenderer:
    def __init__(self, width, height, intrinsic_mat):
        # Initialize Open3D visualizer for 3D rendering
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window()
        
        # Lock to control access to the Open3D visualizer from multiple threads
        self.vis_lock = Lock()
        
        # Open3D Image object for depth processing
        self.width = width
        self.height = height
        self.intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width, height, intrinsic_mat[0, 0], intrinsic_mat[1, 1], intrinsic_mat[0, 2], intrinsic_mat[1, 2]
        )

    def extract_closest_object(self, detections, depth_frame):
        # Initialize variables for tracking the closest object
        closest_obj = None
        min_depth = float('inf')
        
        # Loop through the detected objects and calculate their depths
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            # Get the average depth within the bounding box of the detected object
            obj_depth = np.mean(depth_frame[y1:y2, x1:x2])
            if obj_depth > 0 and obj_depth < min_depth:
                min_depth = obj_depth
                closest_obj = det
        
        return closest_obj

    def render_object(self, color_frame, depth_frame, closest_obj):
        if closest_obj is None:
            print("No object detected for rendering.")
            return
        
        # Convert depth image to Open3D format
        depth_img = o3d.geometry.Image(depth_frame)
        
        # Create point cloud from depth image
        pcd = o3d.geometry.PointCloud.create_from_depth_image(
            depth_img, self.intrinsic
        )
        
        # Lock the visualizer and update the 3D render
        with self.vis_lock:
            self.vis.clear_geometries()
            self.vis.add_geometry(pcd)
            self.vis.poll_events()
            self.vis.update_renderer()

    def close_visualizer(self):
        # Destroy the Open3D window after rendering is complete
        self.vis.destroy_window()

# Initialize video capture (from webcam)
cap = cv2.VideoCapture(1)

# Initialize the detector, depth estimator, and renderer
intrinsic_mat = np.array([[525.0, 0, 320], [0, 525.0, 240], [0, 0, 1]])
detector = ObjectDetector()
depth_estimator = DepthEstimator()
renderer = ObjectRenderer(640, 480, intrinsic_mat)

start_time = time.time()  # Start the timer to limit execution time

# Main loop for processing frames
while True:
    ret, frame = cap.read()
    if not ret:
        continue
    
    # Estimate depth for the current frame
    depth = depth_estimator.estimate(frame)
    
    # Perform object detection on the current frame
    detections = detector.detect(frame)
    
    # After 5 seconds, find the closest detected object and render it in 3D
    if time.time() - start_time > 5:
        closest_obj = renderer.extract_closest_object(detections, depth)
        renderer.render_object(frame, depth, closest_obj)
        
        # Release the video capture and close the OpenCV windows
        cap.release()
        cv2.destroyAllWindows()
        break
    
    # Display the original video frame and the depth map
    cv2.imshow('Original', frame)
    cv2.imshow('Depth', depth)
    
    # Wait for the user to press 'q' or 'space' to exit
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == ord(' '):
        cap.release()
        cv2.destroyAllWindows()
        break

# Continue running the Open3D visualizer until the user closes the window
while True:
    renderer.vis.poll_events()
    renderer.vis.update_renderer()
