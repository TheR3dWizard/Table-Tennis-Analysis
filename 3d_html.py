import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
import cv2
import plotly.graph_objects as go

def estimate_depth(image_path, model_name="depth-anything/Depth-Anything-V2-Small-hf"):
    img = Image.open(image_path).convert("RGB")
    w, h = img.size

    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModelForDepthEstimation.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    inputs = processor(images=img, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        depth = outputs.predicted_depth.squeeze().cpu().numpy()

    # Resize to original image shape if needed
    if depth.shape != (h, w):
        depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR)
    # Normalize
    depth = (depth - depth.min()) / (depth.max() - depth.min())
    return np.asarray(img), depth, (w, h)

def depth_to_point_cloud_pixel_coords(rgb, depth, image_size, depth_scale=1499.0):
    """
    Generate a point cloud with X, Y as pixel coordinates and Z from depth map scaled to [0, 1499].
    """
    h_img, w_img = image_size[1], image_size[0]
    u, v = np.meshgrid(np.arange(w_img), np.arange(h_img), indexing='xy')
    x = u.astype(np.float32)  # X: 0 to w_img-1
    y = v.astype(np.float32)  # Y: 0 to h_img-1
    z = (depth * depth_scale).astype(np.float32)  # Z: depth scaled to [0, 1499]
    points = np.column_stack([x.flatten(), y.flatten(), z.flatten()])
    colors = rgb.reshape(-1, 3).astype(np.float32) / 255.0
    return points, colors

def plot_point_cloud_pixel_coords(points, colors, sample_size=100000):
    """
    Plot a point cloud with X, Y as pixel coordinates using Plotly and save as HTML.
    """
    if len(points) > sample_size:
        idx = np.random.choice(len(points), sample_size, replace=False)
        pts, cols = points[idx], colors[idx]
    else:
        pts, cols = points, colors

    rgb_int = (cols * 255).astype(int)
    color_hex = [f'rgb({r},{g},{b})' for r, g, b in rgb_int]

    fig = go.Figure(data=[go.Scatter3d(
        x=pts[:,0], y=pts[:,1], z=pts[:,2],
        mode='markers',
        marker=dict(size=2, color=color_hex, opacity=0.6)
    )])
    fig.update_layout(
        scene=dict(
            xaxis=dict(title='X (pixels)'),
            yaxis=dict(title='Y (pixels)'),
            zaxis=dict(title='Z (depth)'),
            aspectmode='data'  # Use actual data ranges
        ),
        title="3D Point Cloud with Pixel Coordinates"
    )
    fig.show()
    # Save as timestamped HTML file
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"pointcloud_pixel_coords_{timestamp}.html"
    fig.write_html(filename)
    print(f"Saved interactive plot: {filename}")

if __name__ == "__main__":
    image_path = "image.png"  # Change to your image file
    rgb_img, depth_map, img_size = estimate_depth(image_path)
    print(f"Image size: {img_size}  Depth shape: {depth_map.shape}")
    print(f"RGB shape: {rgb_img.shape}")
    points, colors = depth_to_point_cloud_pixel_coords(rgb_img, depth_map, img_size, depth_scale=1499.0)
    print(f"Generated dense point cloud with {points.shape[0]} points")
    plot_point_cloud_pixel_coords(points, colors)