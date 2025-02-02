import cv2
import torch
import numpy as np
import open3d as o3d
import time
from threading import Lock
from queue import Queue
import threading
from concurrent.futures import ThreadPoolExecutor

# Load MiDaS model
model_type = "MiDaS_small"
midas = torch.hub.load("intel-isl/MiDaS", model_type, trust_repo=True)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
midas.to(device)
midas.eval()

if device.type == "cuda":
    midas.half()

midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.small_transform

class FrameProcessor:
    def __init__(self, queue_size=2):
        self.frame_queue = Queue(maxsize=queue_size)
        self.result_queue = Queue(maxsize=queue_size)
        self.is_running = True
        self.processing_thread = threading.Thread(target=self._process_frames)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        self.last_depth = None
        self.smoothing_factor = 0.8  # Increased smoothing

    def _process_frames(self):
        while self.is_running:
            if not self.frame_queue.empty():
                frame = self.frame_queue.get()
                if frame is None:
                    continue
                
                # Extract center region (desk-sized area)
                height, width = frame.shape[:2]
                center_x, center_y = width // 2, height // 2
                crop_width = int(width * 0.6)  # Adjust crop size as needed
                crop_height = int(height * 0.6)
                
                x1 = center_x - crop_width // 2
                x2 = center_x + crop_width // 2
                y1 = center_y - crop_height // 2
                y2 = center_y + crop_height // 2
                
                cropped_frame = frame[y1:y2, x1:x2]
                
                # Process at higher resolution for better detail
                frame_small = cv2.resize(cropped_frame, (384, 288))
                img_rgb = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)
                
                depth = self.apply_depth_estimation(img_rgb)
                
                if depth is not None:
                    if self.last_depth is not None:
                        depth = self.smoothing_factor * self.last_depth + (1 - self.smoothing_factor) * depth
                    self.last_depth = depth

                    # Enhanced depth processing
                    depth = cv2.bilateralFilter(depth.astype(np.float32), d=7, sigmaColor=0.05, sigmaSpace=7)
                    
                    # Improved normalization for better contrast
                    depth_min = np.percentile(depth, 5)
                    depth_max = np.percentile(depth, 95)
                    depth_normalized = np.clip((depth - depth_min) / (depth_max - depth_min + 1e-6), 0, 1)
                    
                    # Apply adaptive histogram equalization for better detail
                    depth_gray = np.uint8(depth_normalized * 255)
                    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
                    depth_gray = clahe.apply(depth_gray)
                    
                    # Create full-size output with cropped region
                    output = np.zeros((height, width), dtype=np.uint8)
                    output[y1:y2, x1:x2] = cv2.resize(depth_gray, (crop_width, crop_height))
                    
                    # Draw rectangle around focused area
                    output_visual = cv2.cvtColor(output, cv2.COLOR_GRAY2BGR)
                    cv2.rectangle(output_visual, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    self.result_queue.put((output_visual, depth))
            else:
                time.sleep(0.001)

    @torch.inference_mode()
    def apply_depth_estimation(self, img):
        input_batch = transform(img).to(device)
        if device.type == "cuda":
            input_batch = input_batch.half()

        prediction = midas(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bilinear",
            align_corners=False,
        ).squeeze()

        return prediction.cpu().numpy()

    def add_frame(self, frame):
        if not self.frame_queue.full():
            self.frame_queue.put(frame)

    def get_result(self):
        if not self.result_queue.empty():
            return self.result_queue.get()
        return None, None

    def stop(self):
        self.is_running = False
        self.processing_thread.join()

def depth_to_point_cloud(depth_image, color_image, downsample_factor=4):
    try:
        height, width = depth_image.shape
        height_ds = height // downsample_factor
        width_ds = width // downsample_factor
        
        depth_ds = cv2.resize(depth_image, (width_ds, height_ds))
        color_ds = cv2.resize(color_image, (width_ds, height_ds))
        
        # Adjusted camera parameters for desk-sized area
        fx, fy = 400.0 / downsample_factor, 400.0 / downsample_factor
        cx, cy = width_ds // 2, height_ds // 2
        
        y_coords, x_coords = np.mgrid[0:height_ds, 0:width_ds]
        
        # Enhanced depth scaling for better detail
        z = depth_ds / 255.0 * 5  # Reduced depth range for desk-sized area
        
        # More sensitive depth threshold
        mask = z > 0.02
        
        if not np.any(mask):
            return None
            
        x = (x_coords[mask] - cx) * z[mask] / fx
        y = (y_coords[mask] - cy) * z[mask] / fy
        z = z[mask]
        
        points = np.stack((x, y, z), axis=-1)
        colors = color_ds[mask] / 255.0
        
        # Refined outlier filtering
        mean = np.mean(points, axis=0)
        std = np.std(points, axis=0)
        mask = np.all(np.abs(points - mean) <= 2.5 * std, axis=1)
        points = points[mask]
        colors = colors[mask]
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        
        # More aggressive outlier removal for cleaner visualization
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=30, std_ratio=1.5)
        
        # Finer voxel downsampling for more detail
        pcd = pcd.voxel_down_sample(voxel_size=0.01)
        
        return pcd
    except Exception as e:
        print(f"Error in point cloud generation: {e}")
        return None

# PointCloudVisualizer class remains the same
class PointCloudVisualizer:
    def __init__(self):
        self.is_running = False
        self.pcd = None
        self.vis_lock = Lock()
        self.last_update_time = 0
        self.update_interval = 0.1
        self.previous_points = None
        self.smoothing_factor = 0.5

    def update_geometry(self, pcd):
        current_time = time.time()
        if current_time - self.last_update_time < self.update_interval:
            return
        
        with self.vis_lock:
            if self.previous_points is not None and pcd is not None:
                current_points = np.asarray(pcd.points)
                prev_points = np.asarray(self.previous_points.points)
                
                if abs(len(current_points) - len(prev_points)) < len(prev_points) * 0.3:
                    min_points = min(len(current_points), len(prev_points))
                    smoothed_points = (
                        self.smoothing_factor * prev_points[:min_points] +
                        (1 - self.smoothing_factor) * current_points[:min_points]
                    )
                    pcd.points = o3d.utility.Vector3dVector(smoothed_points)
            
            self.pcd = pcd
            self.previous_points = pcd
            self.last_update_time = current_time
            self.is_running = True

    def render_once(self):
        with self.vis_lock:
            if self.pcd is not None:
                vis = o3d.visualization.Visualizer()
                vis.create_window(width=800, height=600)
                opt = vis.get_render_option()
                opt.point_size = 2.5  # Slightly larger points
                opt.background_color = np.asarray([0, 0, 0])
                
                vis.add_geometry(self.pcd)
                vis.run()
                vis.destroy_window()

    def close(self):
        self.is_running = False

def capture_and_process():
    cap = cv2.VideoCapture(1)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # Higher resolution input
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print("Error: Failed to open camera")
        return

    frame_processor = FrameProcessor()
    visualizer = PointCloudVisualizer()
    executor = ThreadPoolExecutor(max_workers=1)

    cv2.namedWindow('Depth', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Depth', 640, 480)
    
    show_point_cloud = False
    last_fps_time = time.time()
    frame_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_processor.add_frame(frame)
            depth_result, raw_depth = frame_processor.get_result()
            
            if depth_result is not None:
                frame_count += 1
                current_time = time.time()
                if current_time - last_fps_time >= 1.0:
                    fps = frame_count / (current_time - last_fps_time)
                    print(f"FPS: {fps:.2f}")
                    frame_count = 0
                    last_fps_time = current_time

                cv2.imshow('Depth', depth_result)

                if show_point_cloud:
                    future_pcd = executor.submit(depth_to_point_cloud, raw_depth, frame)
                    pcd = future_pcd.result()
                    if pcd is not None:
                        visualizer.update_geometry(pcd)
                        visualizer.render_once()

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('v'):
                show_point_cloud = not show_point_cloud
                print(f"Point cloud visualization: {'On' if show_point_cloud else 'Off'}")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        frame_processor.stop()
        visualizer.close()

if __name__ == "__main__":
    capture_and_process()