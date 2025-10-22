import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
import cv2
import plotly.graph_objects as go
import h5py

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
    depth = (depth - depth.min()) / (depth.max() - depth.min())
    return np.asarray(img), depth, (w, h)

def depth_to_point_cloud_scale_xy(rgb, depth, image_size, xy_scale=1.5, depth_scale=1.0):
    h_img, w_img = image_size[1], image_size[0]
    u, v = np.meshgrid(np.arange(w_img), np.arange(h_img), indexing='xy')
    x = ((u - w_img/2) * xy_scale).astype(np.float32)
    y = ((v - h_img/2) * xy_scale).astype(np.float32)
    z = (depth * depth_scale).astype(np.float32)
    points = np.column_stack([x.flatten(), y.flatten(), z.flatten()])
    colors = rgb.reshape(-1, 3).astype(np.float32) / 255.0
    return points, colors

def make_pointcloud_struct(points, colors):
    N = points.shape[0]
    dtype = np.dtype([
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('r', 'f4'), ('g', 'f4'), ('b', 'f4')
    ])
    arr = np.empty(N, dtype=dtype)
    arr['x'], arr['y'], arr['z'] = points[:,0], points[:,1], points[:,2]
    arr['r'], arr['g'], arr['b'] = colors[:,0], colors[:,1], colors[:,2]
    return arr

def plot_point_cloud_xy_scale(points, colors, sample_size=75000):
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
            xaxis=dict(title='X (scaled by 1.5)'),
            yaxis=dict(title='Y (scaled by 1.5)'),
            zaxis=dict(title='Z (normalized depth)'),
            aspectmode='cube'
        ),
        title="3D Point Cloud (X and Y scaled by 1.5)"
    )
    fig.show()
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"pointcloud_xy_scaled_{timestamp}.html"
    fig.write_html(filename)
    print(f"Saved interactive plot: {filename}")

if __name__ == "__main__":
    image_path = "image.png"  # Change to your image file
    rgb_img, depth_map, img_size = estimate_depth(image_path)
    print(f"Image size: {img_size}  Depth shape: {depth_map.shape}")
    points, colors = depth_to_point_cloud_scale_xy(rgb_img, depth_map, img_size, xy_scale=1.5, depth_scale=1.0)
    print(f"Generated dense point cloud with {points.shape[0]} points")
    
    # Visualization
    plot_point_cloud_xy_scale(points, colors)
    
    # Efficient export as npz and h5
    pc_struct = make_pointcloud_struct(points, colors)
    np.savez_compressed('pointcloud_struct.npz', pc=pc_struct)
    print("Saved point cloud as pointcloud_struct.npz (compressed npz)")
    
    # with h5py.File('pointcloud_struct.h5', 'w') as f:
    #     f.create_dataset('pointcloud', data=pc_struct, compression='gzip')
    # print("Saved point cloud as pointcloud_struct.h5 (HDF5)")

    # You can reload via:
    # data = np.load('pointcloud_struct.npz')['pc']
    # or
    # with h5py.File('pointcloud_struct.h5','r') as f: pc_struct = f['pointcloud'][:]
