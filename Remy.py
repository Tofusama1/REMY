import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from PIL import Image
import torch
from transformers import GLPNImageProcessor, GLPNForDepthEstimation
import cv2
import time
import numpy as np
import open3d as o3d

def capture_from_iphone():
    # Initialize camera
    cap = cv2.VideoCapture(1)  # Use 0 for default camera
    
    if not cap.isOpened():
        raise RuntimeError("Could not open camera")
    
    print("Camera initialized. Get ready!")
    # Countdown
    for i in range(5, 0, -1):
        print(f"Taking photo in {i} seconds...")
        time.sleep(1)
    
    print("SNAP!")
    
    # Capture frame
    ret, frame = cap.read()
    if not ret:
        cap.release()
        raise RuntimeError("Could not capture image")
    
    # Convert from BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Convert to PIL Image
    image = Image.fromarray(rgb_frame)
    
    # Release camera
    cap.release()
    
    return image

def process_depth_estimation(image):
    # Load the GLPN model
    feature_extractor = GLPNImageProcessor.from_pretrained("vinvino02/glpn-nyu")
    model = GLPNForDepthEstimation.from_pretrained("vinvino02/glpn-nyu")
    
    # Resize image
    new_height = 480 if image.height > 480 else image.height
    new_height -= (new_height % 32)
    new_width = int(new_height * image.width / image.height)
    diff = new_width % 32
    new_width = new_width - diff if diff < 16 else new_width + (32 - diff)
    new_size = (new_width, new_height)
    image = image.resize(new_size)
    
    # Preprocess and perform depth estimation
    inputs = feature_extractor(images=image, return_tensors="pt")
    with torch.no_grad():
        output = model(**inputs)
        predicted_depth = output.predicted_depth
    
    # Process depth map
    pad = 16
    output = predicted_depth.squeeze().cpu().numpy() * 1000.0
    output = output[pad:-pad, pad:-pad]
    image = image.crop((pad, pad, new_width - pad, new_height - pad))
    
    return image, output

def create_and_display_3d(image, output):
    # Prepare depth image
    width, height = image.size
    depth_image = (output * 255 / np.max(output)).astype('uint8')
    image_array = np.array(image)
    
    # Create RGBD image
    depth_o3d = o3d.geometry.Image(depth_image)
    image_o3d = o3d.geometry.Image(image_array)
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        image_o3d, depth_o3d, convert_rgb_to_intensity=False
    )
    
    # Set camera parameters
    camera_intrinsic = o3d.camera.PinholeCameraIntrinsic()
    camera_intrinsic.set_intrinsics(width, height, 500, 500, width/2, height/2)
    
    # Create and process point cloud
    pcd_raw = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, camera_intrinsic)
    cl, ind = pcd_raw.remove_statistical_outlier(nb_neighbors=20, std_ratio=6.0)
    pcd = pcd_raw.select_by_index(ind)
    
    # Process mesh
    pcd.estimate_normals()
    pcd.orient_normals_to_align_with_direction()
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=10, n_threads=1)[0]
    rotation = mesh.get_rotation_matrix_from_xyz((np.pi, 0, 0))
    mesh.rotate(rotation, center=(0, 0, 0))
    
    # Create final colored mesh
    mesh_uniform = mesh.paint_uniform_color([0.9, 0.8, 0.9])
    mesh_uniform.compute_vertex_normals()
    
    # Display results
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(image)
    ax[0].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    ax[1].imshow(output, cmap='plasma')
    ax[1].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    plt.tight_layout()
    plt.show()
    
    # Display 3D visualizations
    o3d.visualization.draw_geometries([pcd])
    o3d.visualization.draw_geometries([mesh], mesh_show_wireframe=True)
    o3d.visualization.draw_geometries([mesh_uniform], mesh_show_back_face=True)
    o3d.io.write_triangle_mesh("output_mesh.ply", mesh_uniform)
    print("Mesh saved as output_mesh.ply")


def main():
    try:
        # Capture image from iPhone camera
        image = capture_from_iphone()
        
        # Process depth estimation
        processed_image, depth_output = process_depth_estimation(image)
        
        # Create and display 3D visualization
        create_and_display_3d(processed_image, depth_output)
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
