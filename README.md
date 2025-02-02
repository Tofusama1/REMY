# REMY
***
Repository for Rutgers HackRU 2025

## Overview
***
Our project transforms interior design planning by using AI and computer vision to create instant 3D models of rooms using just a phone camera. The system captures a room photo, analyzes its depth using neural networks, and generates a detailed 3D model. This allows users to accurately measure their space and virtually experiment with furniture placement and room layouts, making interior design more accessible and efficient.

## Features
***
** Image Capture **
- 5-second countdown timer for stable photo capture.
- Works with iPhone camera via IP camera connection.
- Real-time terminal feedback during the capture process.
  
**Depth Analysis**
- AI-powered depth estimation for each pixel.
- Processes standard room photos into depth maps.
- Visual representation using a color-mapped depth display.
- Side-by-side view of the original image and depth map.

**3D Reconstruction**
- Converts depth data into a point cloud representation.
- Removes statistical outliers for cleaner 3D models.
- Generates textured 3D mesh from the point cloud.
- Creates wireframe view for structural analysis.
- Produces a final colored 3D model with a uniform surface.

**Visualization**
- Interactive 3D model viewer.
- Multiple viewing modes:
  - Point cloud view.
  - Wireframe mesh view.
  - Solid colored mesh view.
- Real-time model rotation and zoom capabilities.

**Error Handling**
- Camera connection validation.
- Capture failure detection.
- Process monitoring throughout the pipeline.
- User-friendly error messages.

## Purpose
***
This project aims to revolutionize interior design planning by making it more accessible and efficient for everyday users. The main purpose is to transform a simple phone camera into a powerful 3D room scanning tool that eliminates common frustrations in room planning and furniture buying. The ultimate goal is to help users make better-informed decisions about their living spaces while saving time, reducing costs, and minimizing the stress typically associated with interior design projects.

## Tech Stack Used
***
- **Programming Language:**
  - Pyton
- **Libraries & Frameworks:**
  - OpenCV (cv2)
  - NumPy
  - Matplotlib
  - PIL (Pillow)
  - PyTorch
  - Transformers (Hugging Face)
  - Open3D
  - Time
## Installation & Setup
***
1. Clone the repository
   ```bash
   git clone https://github.com/your-repo/REMY.git
   ```
   ```bash
   cd REMY
   ```
2. Install dependencies
   ```bash
   pip install -r requirement.txt
   ```
4. Run the application
   ```bash
   python main.py
   ```

## GitHub Setup for New Users

If you’re new to GitHub, follow these steps to set up the repository locally:

1. **Create a GitHub Account**: Go to [GitHub](https://github.com/) and sign up if you don’t already have an account.

2. **Install Git**: 
   - On Windows, download and install Git from [here](https://git-scm.com/download/win).
   - On macOS, use Homebrew by running `brew install git`.
   - On Linux, use your package manager, e.g., `sudo apt install git` for Ubuntu.

3. **Clone the Repository**:
   - Open your terminal or command prompt.
   - Run the following command to clone the repository:
     ```bash
     git clone https://github.com/YourUsername/REMY.git
     ```
   - Replace `YourUsername` with the GitHub username associated with the repository if it’s hosted on your account.

4. **Navigate to the Project Directory**:
   - Go to the directory where you cloned the repo:
     ```bash
     cd REMY
     ```

5. **Setting Up Git Remotes**:
   - If you plan to make changes and push them, add a remote origin:
     ```bash
     git remote add origin https://github.com/YourUsername/REMY.git
     ```

6. **Sync Changes**:
   - To pull any new changes from the repository, run:
     ```bash
     git pull origin main
     ```
