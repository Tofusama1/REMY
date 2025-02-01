import os
from flask import Flask, request, jsonify
import cv2
import open3d as o3d


# # Install system dependencies for OpenCV and Open3D
# sudo apt-get update && sudo apt-get install -y libgl1-mesa-glx

# # Install Python dependencies
# pip install opencv-python open3d flask


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './uploads'

def detect_edges(image_path):
    # Read the image and convert to grayscale.
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not read image at {}".format(image_path))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    return edges

def depth_to_point_cloud(depth_path):
    # Dummy implementation: create an empty point cloud.
    # Replace with actual depth-to-point-cloud conversion logic.
    pcd = o3d.geometry.PointCloud()
    return pcd

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'image_file' not in request.files or 'depth_file' not in request.files:
        return jsonify({"message": "Missing file parameter"}), 400
    
    image_file = request.files['image_file']
    depth_file = request.files['depth_file']
    
    if image_file.filename == '' or depth_file.filename == '':
        return jsonify({"message": "No selected file"}), 400

    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_file.filename)
    depth_path = os.path.join(app.config['UPLOAD_FOLDER'], depth_file.filename)
    
    image_file.save(image_path)
    depth_file.save(depth_path)
    
    try:
        edges = detect_edges(image_path)
        edges_path = image_path.replace(".png", "_edges.png")
        cv2.imwrite(edges_path, edges)
    except Exception as e:
        return jsonify({"message": "Error processing image: {}".format(e)}), 500

    try:
        pcd = depth_to_point_cloud(depth_path)
        pcd_path = depth_path.replace(".png", ".ply")
        o3d.io.write_point_cloud(pcd_path, pcd)
    except Exception as e:
        return jsonify({"message": "Error processing depth file: {}".format(e)}), 500

    return jsonify({"message": "Files processed successfully", "pcd_path": pcd_path}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)