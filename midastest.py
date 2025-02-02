import cv2
import torch
import numpy as np
import open3d as o3d
import time
from threading import Lock
from concurrent.futures import ThreadPoolExecutor

# Load MiDaS model
model_type = "DPT_Large"
midas = torch.hub.load("intel-isl/MiDaS", model_type, trust_repo=True)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
midas.to(device)
midas.eval()

if device.type == "cuda":
    midas.half()  # Use half precision for faster inference

midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.dpt_transform

# Point Cloud Visualization Class
class PointCloudVisualizer:
    def __init__(self):
        self.is_running = False
        self.pcd = None
        self.vis_lock = Lock()
        self.window_size = (800, 600)
        self.last_render_time = 0
        self.render_interval = 0.2  # Update every 200ms

    def update_geometry(self, pcd):
        if pcd is None:
            return
        with self.vis_lock:
            self.pcd = o3d.geometry.PointCloud(pcd)
            self.is_running = True
    
    def render_once(self):
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
                    )
                    self.last_render_time = current_time
                except Exception as e:
                    print(f"Render error: {e}")

    def close(self):
        self.is_running = False
        with self.vis_lock:
            self.pcd = None

# Capture and Process Video Feed
def capture_and_process():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, 30)  # Set FPS to 30 for smoother frames
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("Error: Failed to open camera")
        return

    visualizer = PointCloudVisualizer()
    executor = ThreadPoolExecutor(max_workers=3)  # More threads for better parallelism

    cv2.namedWindow('Depth', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Depth', 640, 480)
    
    show_point_cloud = False

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame")
            break

        frame_resized = cv2.resize(frame, (320, 240))  # Reduce size before processing
        img_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

        # Apply depth estimation asynchronously
        future = executor.submit(apply_depth_estimation, img_rgb)
        depth = future.result()

        if depth is not None:
            # Normalize depth for proper grayscale visualization
            depth_min = np.min(depth)
            depth_max = np.max(depth)
            depth_normalized = (depth - depth_min) / (depth_max - depth_min + 1e-6)  # Avoid division by zero
            depth_gray = np.uint8(depth_normalized * 255)

            # Display grayscale depth map
            cv2.imshow('Depth', depth_gray)

            # Convert depth to point cloud less frequently
            if show_point_cloud and time.time() - visualizer.last_render_time > 0.2:
                future_pcd = executor.submit(depth_to_point_cloud, depth, frame)
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

    cap.release()
    cv2.destroyAllWindows()
    visualizer.close()

# Apply depth estimation
def apply_depth_estimation(img):
    input_batch = transform(img).to(device)
    if device.type == "cuda":
        input_batch = input_batch.half()  # Convert to half precision if on CUDA

    with torch.no_grad():
        prediction = midas(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    return prediction.cpu().numpy()

# Depth to Point Cloud function
def depth_to_point_cloud(depth_image, color_image, downsample_factor=4):
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

if __name__ == "__main__":
    capture_and_process()