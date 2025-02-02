import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from PIL import Image
import torch
import cv2
import time
import numpy as np
import open3d as o3d
from transformers import DPTImageProcessor, DPTForDepthEstimation

def capture_from_iphone():
    cap = cv2.VideoCapture(1)  # Use 0 for default camera
    if not cap.isOpened():
        raise RuntimeError("Could not open camera")
    
    print("Camera initialized. Get ready!")
    for i in range(5, 0, -1):
        print(f"Taking photo in {i} seconds...")
        time.sleep(1)
    
    print("SNAP!")
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError("Could not capture image")
    
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

def process_depth_estimation(image):
    feature_extractor = DPTImageProcessor.from_pretrained("Intel/dpt-large")
    model = DPTForDepthEstimation.from_pretrained("Intel/dpt-large")
    
    new_height = 480 if image.height > 480 else image.height
    new_height -= (new_height % 32)
    new_width = int(new_height * image.width / image.height)
    diff = new_width % 32
    new_width = new_width - diff if diff < 16 else new_width + (32 - diff)
    
    image = image.resize((new_width, new_height))
    inputs = feature_extractor(images=image, return_tensors="pt")
    
    with torch.no_grad():
        predicted_depth = model(**inputs).predicted_depth
    
    depth_output = predicted_depth.squeeze().cpu().numpy() * 1000.0
    depth_output = cv2.resize(depth_output, (new_width, new_height))  # Resize depth to match RGB
    
    depth_output = cv2.bilateralFilter(depth_output.astype(np.float32), 9, 75, 75)
    
    return image, depth_output

def create_and_display_3d(image, depth_output):
    width, height = image.size
    depth_image = (depth_output * 255 / np.max(depth_output)).astype('uint8')
    image_array = np.array(image)
    
    depth_o3d = o3d.geometry.Image(depth_image)
    image_o3d = o3d.geometry.Image(image_array)
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        image_o3d, depth_o3d, convert_rgb_to_intensity=False
    )
    
    camera_intrinsic = o3d.camera.PinholeCameraIntrinsic()
    camera_intrinsic.set_intrinsics(width, height, 500, 500, width/2, height/2)
    
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, camera_intrinsic)
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=50, std_ratio=2.0)
    pcd = pcd.select_by_index(ind)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=5, max_nn=30))
    
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=12, linear_fit=True)[0]
    mesh.rotate(mesh.get_rotation_matrix_from_xyz((np.pi, 0, 0)), center=(0, 0, 0))
    
    mesh_uniform = mesh.paint_uniform_color([0.9, 0.8, 0.9])
    mesh_uniform.compute_vertex_normals()
    
    o3d.io.write_point_cloud("output.pcd", pcd)  # Export point cloud
    
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(image)
    ax[0].axis('off')
    ax[1].imshow(depth_output, cmap='plasma')
    ax[1].axis('off')
    plt.show()
    
    o3d.visualization.draw_geometries([pcd])
    o3d.visualization.draw_geometries([mesh], mesh_show_wireframe=True)
    o3d.visualization.draw_geometries([mesh_uniform], mesh_show_back_face=True)

def main():
    try:
        image = capture_from_iphone()
        processed_image, depth_output = process_depth_estimation(image)
        create_and_display_3d(processed_image, depth_output)
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
