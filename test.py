import os
import numpy as np
import cv2
import open3d as o3d
from flask import Flask, request, jsonify
import threading


#What you need to install:
# pip install opencv-python
# sudo apt-get update
# sudo apt-get install -y libgl1-mesa-glx
# pip install open3d
# sudo apt-get install libosmesa6-dev
# sudo apt-get install -y libglew-dev libglfw3-dev
# export XDG_RUNTIME_DIR=/tmp/runtime-yourusername

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './uploads'

def detect_edges(image):
    """Optimized edge detection using Canny."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    # Convert back to color for visualization
    edges_color = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    edges_color[edges > 0] = [0, 255, 255]  # Yellow edges
    return edges_color

def estimate_depth_from_image(image):
    """Optimized depth estimation with colormap."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    
    magnitude = np.sqrt(np.square(sobelx) + np.square(sobely))
    depth = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    depth = np.uint8(depth)
    
    # Apply colormap for visualization
    depth_colormap = cv2.applyColorMap(depth, cv2.COLORMAP_JET)
    return depth, depth_colormap

def depth_to_point_cloud(depth_image, color_image):
    """Optimized point cloud generation with color."""
    height, width = depth_image.shape
    fx, fy = 525.0, 525.0
    cx, cy = width // 2, height // 2
    
    # Create coordinate grids
    y_coords, x_coords = np.mgrid[0:height, 0:width]
    
    # Vectorized depth processing
    z = depth_image / 255.0  # Normalize depth
    
    # Calculate x and y coordinates for all points at once
    x = (x_coords - cx) * z / fx
    y = (y_coords - cy) * z / fy
    
    # Stack coordinates and reshape
    mask = z > 0
    points = np.stack((x[mask], y[mask], z[mask]), axis=-1)
    
    # Get colors for valid points
    colors = color_image[mask] / 255.0
    
    # Create and color the point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    return pcd

def capture_and_visualize():
    """Optimized capture and visualization with color."""
    cap = cv2.VideoCapture(1)
    
    # Set higher resolution for camera capture
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    # Set up Open3D visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window("Point Cloud Visualization", width=1280, height=720)
    
    # Optimize rendering
    opt = vis.get_render_option()
    opt.point_size = 3
    opt.background_color = np.asarray([0, 0, 0])
    
    # Initialize view control
    ctr = vis.get_view_control()
    
    # Create named windows
    cv2.namedWindow('Edge Detection', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Estimated Depth', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Original', cv2.WINDOW_NORMAL)
    
    # Resize windows
    cv2.resizeWindow('Edge Detection', 1280, 720)
    cv2.resizeWindow('Estimated Depth', 1280, 720)
    cv2.resizeWindow('Original', 1280, 720)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame")
            break
        
        # Process frames
        edges_color = detect_edges(frame)
        depth, depth_colormap = estimate_depth_from_image(frame)
        
        # Prepare color image for point cloud
        color_for_cloud = cv2.applyColorMap(depth, cv2.COLORMAP_RAINBOW)
        
        # Show processed images
        cv2.imshow('Edge Detection', edges_color)
        cv2.imshow('Estimated Depth', depth_colormap)
        cv2.imshow('Original', frame)
        
        try:
            pcd = depth_to_point_cloud(depth, color_for_cloud)
            
            # Update visualization
            vis.clear_geometries()
            vis.add_geometry(pcd, reset_bounding_box=False)
            vis.poll_events()
            vis.update_renderer()
            
        except Exception as e:
            print(f"Error processing depth frame: {e}")
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    vis.destroy_window()

if __name__ == '__main__':
    capture_and_visualize()
    app.run(host='0.0.0.0', port=5001, debug=True)
