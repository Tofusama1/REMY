import os
import numpy as np
import cv2
import open3d as o3d
from threading import Thread, Lock
import time
from ultralytics import YOLO
import torch

class ObjectDetector:
    def __init__(self):
        self.model = YOLO('yolov8n.pt')
        self.classes = self.model.names
        
    def detect(self, frame):
        results = self.model(frame, stream=True)
        detections = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                confidence = box.conf[0]
                class_id = box.cls[0]
                detections.append({
                    'bbox': (int(x1), int(y1), int(x2), int(y2)),
                    'confidence': float(confidence),
                    'class': self.classes[int(class_id)]
                })
        return detections

class DepthEstimator:
    def __init__(self):
        self.last_depth = None
        
    def estimate(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(np.square(sobelx) + np.square(sobely))
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        depth = magnitude + np.abs(laplacian)
        depth = cv2.normalize(depth, None, 0, 65535, cv2.NORM_MINMAX)
        depth = np.uint16(depth)
        return depth

class ObjectRenderer:
    def __init__(self):
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window()
        self.vis_lock = Lock()

    def extract_closest_object(self, detections, depth_frame):
        closest_obj = None
        min_depth = float('inf')
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            obj_depth = np.mean(depth_frame[y1:y2, x1:x2])
            if obj_depth > 0 and obj_depth < min_depth:
                min_depth = obj_depth
                closest_obj = det
        
        return closest_obj

    def render_object(self, color_frame, depth_frame, closest_obj):
        if closest_obj is None:
            print("No object detected for rendering.")
            return
        
        x1, y1, x2, y2 = closest_obj['bbox']
        depth_segment = depth_frame[y1:y2, x1:x2]
        color_segment = color_frame[y1:y2, x1:x2]
        
        height, width = depth_segment.shape
        fx, fy = 525.0, 525.0
        cx, cy = width // 2, height // 2
        
        y_coords, x_coords = np.mgrid[0:height, 0:width]
        z = depth_segment.astype(float) / 65535.0 * 5.0
        mask = (z > 0) & (z < 5.0)
        
        if not np.any(mask):
            return
        
        x = (x_coords[mask] - cx) * z[mask] / fx
        y = (y_coords[mask] - cy) * z[mask] / fy
        z = z[mask]
        
        points = np.stack((x, y, z), axis=-1)
        colors = color_segment[mask] / 255.0
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        
        pcd = pcd.voxel_down_sample(voxel_size=0.005)
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        
        with self.vis_lock:
            self.vis.clear_geometries()
            self.vis.add_geometry(pcd)
            self.vis.poll_events()
            self.vis.update_renderer()

    def close_visualizer(self):
        self.vis.destroy_window()

cap = cv2.VideoCapture(1)
detector = ObjectDetector()
depth_estimator = DepthEstimator()
renderer = ObjectRenderer()

start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        continue
    
    depth = depth_estimator.estimate(frame)
    detections = detector.detect(frame)
    
    if time.time() - start_time > 5:
        closest_obj = renderer.extract_closest_object(detections, depth)
        renderer.render_object(frame, depth, closest_obj)
        cap.release()
        cv2.destroyAllWindows()
        break
    
    cv2.imshow('Original', frame)
    cv2.imshow('Depth', depth)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == ord(' '):
        cap.release()
        cv2.destroyAllWindows()
        break

while True:
    renderer.vis.poll_events()
    renderer.vis.update_renderer()
