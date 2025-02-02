import os
import numpy as np
import cv2
import open3d as o3d
from flask import Flask, request, jsonify
from threading import Thread, Lock
import time

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './uploads'

def detect_edges(image):
    """Optimized edge detection using Canny with reduced resolution."""
    small_image = cv2.resize(image, (640, 480))
    gray = cv2.cvtColor(small_image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    edges_color = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    edges_color[edges > 0] = [0, 255, 255]
    return edges_color

def estimate_depth_from_image(image, clipping_distance_meters=5):
    """Optimized depth estimation with reduced computation."""
    try:
        small_image = cv2.resize(image, (640, 480))
        gray = cv2.cvtColor(small_image, cv2.COLOR_BGR2GRAY)
        
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        magnitude = np.sqrt(np.square(sobelx) + np.square(sobely))
        depth = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
        depth = np.uint8(depth)
        
        clipping_distance = int(clipping_distance_meters * 255)
        mask = depth > clipping_distance
        
        depth_colormap = cv2.applyColorMap(depth, cv2.COLORMAP_JET)
        
        return depth, depth_colormap, mask
    except Exception as e:
        print(f"Error in depth estimation: {e}")
        return None, None, None

def depth_to_point_cloud(depth_image, color_image, downsample_factor=4):
    """Optimized point cloud generation with downsampling and error handling."""
    try:
        height, width = depth_image.shape
        height_ds = height // downsample_factor
        width_ds = width // downsample_factor
        
        depth_ds = cv2.resize(depth_image, (width_ds, height_ds))
        color_ds = cv2.resize(color_image, (width_ds, height_ds))
        
        fx, fy = 525.0 / downsample_factor, 525.0 / downsample_factor
        cx, cy = width_ds // 2, height_ds // 2
        
        y_coords, x_coords = np.mgrid[0:height_ds, 0:width_ds]
        
        z = depth_ds / 255.0
        mask = z > 0
        
        if not np.any(mask):
            return None
            
        x = (x_coords[mask] - cx) * z[mask] / fx
        y = (y_coords[mask] - cy) * z[mask] / fy
        z = z[mask]
        
        points = np.stack((x, y, z), axis=-1)
        colors = color_ds[mask] / 255.0
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        
        pcd = pcd.voxel_down_sample(voxel_size=0.01)
        
        return pcd
    except Exception as e:
        print(f"Error in point cloud generation: {e}")
        return None

class PointCloudVisualizer:
    def __init__(self):
        self.is_running = False
        self.pcd = None
        self.vis_lock = Lock()
        self.window_size = (800, 600)
        self.last_render_time = 0
        self.render_interval = 0.1  # 100ms between renders
        
    def update_geometry(self, pcd):
        """Update point cloud data without rendering."""
        if pcd is None:
            return
            
        with self.vis_lock:
            self.pcd = o3d.geometry.PointCloud(pcd)
            self.is_running = True
    
    def render_once(self):
        """Render current point cloud in a new window."""
        current_time = time.time()
        if current_time - self.last_render_time < self.render_interval:
            return
            
        with self.vis_lock:
            if self.pcd is not None:
                try:
                    o3d.visualization.draw_geometries(
                        [self.pcd],
                        window_name="Point Cloud View",
                        width=self.window_size[0],
                        height=self.window_size[1],
                        left=50,
                        top=50,
                        point_show_normal=False,
                        mesh_show_wireframe=False,
                        mesh_show_back_face=False
                    )
                    self.last_render_time = current_time
                except Exception as e:
                    print(f"Render error: {e}")
    
    def close(self):
        """Clean up resources."""
        self.is_running = False
        with self.vis_lock:
            self.pcd = None

def save_point_cloud(pcd, filename="out.ply"):
    """Save point cloud to PLY file with error handling."""
    try:
        if pcd is not None:
            o3d.io.write_point_cloud(filename, pcd, write_ascii=True)
            print(f"Point cloud saved to {filename}")
        else:
            print("Error: No point cloud data to save")
    except Exception as e:
        print(f"Error saving point cloud: {e}")

def capture_and_visualize():
    """Capture and visualization with improved resource management."""
    cap = None
    visualizer = None
    
    try:
        # Initialize camera
        cap = cv2.VideoCapture(1)
        if not cap.isOpened():
            raise Exception("Failed to open camera")
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        # Initialize visualizer
        visualizer = PointCloudVisualizer()
        
        # Create windows
        cv2.namedWindow('Original', cv2.WINDOW_NORMAL)
        cv2.namedWindow('Depth', cv2.WINDOW_NORMAL)
        cv2.namedWindow('Edge Detection', cv2.WINDOW_NORMAL)
        
        # Position windows
        cv2.moveWindow('Original', 0, 0)
        cv2.moveWindow('Depth', 640, 0)
        cv2.moveWindow('Edge Detection', 1280, 0)
        
        # Size windows
        cv2.resizeWindow('Original', 640, 480)
        cv2.resizeWindow('Depth', 640, 480)
        cv2.resizeWindow('Edge Detection', 640, 480)
        
        last_point_cloud_update = time.time()
        point_cloud_update_interval = 0.1
        show_point_cloud = False  # Flag to control point cloud visualization
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture frame")
                break
            
            # Process frame
            edges_color = detect_edges(frame)
            depth, depth_colormap, mask = estimate_depth_from_image(frame)
            
            if depth_colormap is not None:
                # Show 2D visualizations
                cv2.imshow('Original', frame)
                cv2.imshow('Depth', depth_colormap)
                cv2.imshow('Edge Detection', edges_color)
                
                # Update point cloud at intervals
                current_time = time.time()
                if current_time - last_point_cloud_update > point_cloud_update_interval:
                    try:
                        if depth is not None:
                            pcd = depth_to_point_cloud(depth, frame)
                            if pcd is not None:
                                visualizer.update_geometry(pcd)
                                if show_point_cloud:
                                    visualizer.render_once()
                            last_point_cloud_update = current_time
                    except Exception as e:
                        print(f"Error processing depth frame: {e}")
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('e'):
                if visualizer.pcd is not None:
                    save_point_cloud(visualizer.pcd)
            elif key == ord('v'):
                show_point_cloud = not show_point_cloud
                print(f"Point cloud visualization: {'On' if show_point_cloud else 'Off'}")
            elif key == ord('p'):
                if os.path.exists("out.ply"):
                    try:
                        saved_cloud = o3d.io.read_point_cloud("out.ply")
                        o3d.visualization.draw_geometries([saved_cloud])
                    except Exception as e:
                        print(f"Error viewing saved point cloud: {e}")
    
    except Exception as e:
        print(f"Fatal error in capture_and_visualize: {e}")
    
    finally:
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()
        if visualizer is not None:
            visualizer.close()

if __name__ == '__main__':
    try:
        capture_and_visualize()
    except Exception as e:
        print(f"Application error: {e}")
    finally:
        app.run(host='0.0.0.0', port=5001, debug=True)