
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
import cv2
from pathlib import Path
import hashlib


class DepthEstimator:
    """
    High-performance depth estimation class with caching for fast repeated queries.
    Uses Depth Anything V2 for accurate depth estimation.
    """
    
    def __init__(self, model_size="large", depth_scale=1499.0, use_cache=True):
        """
        Initialize the DepthEstimator.
        
        Args:
            model_size: 'small', 'base', or 'large' (larger = more accurate but slower)
                       - small: fastest, ~100ms inference
                       - base: balanced, ~200ms inference  
                       - large: most accurate, ~400ms inference
            depth_scale: Scale factor for depth values (default 1499.0)
            use_cache: Cache depth maps for repeated queries on same image
        """
        self.depth_scale = depth_scale
        self.use_cache = use_cache
        self.cache = {}  # Cache for depth maps
        
        # Model selection for accuracy vs speed tradeoff
        model_names = {
            "small": "depth-anything/Depth-Anything-V2-Small-hf",
            "base": "depth-anything/Depth-Anything-V2-Base-hf",
            "large": "depth-anything/Depth-Anything-V2-Large-hf"
        }
        
        self.model_name = model_names.get(model_size.lower(), model_names["large"])
        
        # Initialize model and processor
        print(f"Loading Depth Anything V2 ({model_size}) model...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoImageProcessor.from_pretrained(self.model_name)
        self.model = AutoModelForDepthEstimation.from_pretrained(self.model_name)
        self.model.to(self.device).eval()
        
        # Enable optimizations
        if self.device.type == "cuda":
            self.model = self.model.half()  # Use FP16 for faster inference on GPU
            torch.backends.cudnn.benchmark = True
        
        print(f"Model loaded on {self.device}")
    
    def _get_image_hash(self, image_path):
        """Generate hash of image for caching."""
        with open(image_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    
    def _estimate_depth_map(self, image_path):
        """
        Estimate depth map from image with caching.
        
        Args:
            image_path: Path to input image
            
        Returns:
            depth_map: 2D numpy array of depth values
            img_size: (width, height) tuple
        """
        # Check cache first
        if self.use_cache:
            img_hash = self._get_image_hash(image_path)
            if img_hash in self.cache:
                return self.cache[img_hash]
        
        # Load and process image
        img = Image.open(image_path).convert("RGB")
        w, h = img.size
        
        # Prepare inputs
        inputs = self.processor(images=img, return_tensors="pt")
        
        # Move to device and convert to half precision if using GPU
        if self.device.type == "cuda":
            inputs = {k: v.half().to(self.device) for k, v in inputs.items()}
        else:
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Inference with no gradient computation
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=(self.device.type == "cuda")):
                outputs = self.model(**inputs)
                depth = outputs.predicted_depth.squeeze().cpu().float().numpy()
        
        # Resize to original image dimensions if needed
        if depth.shape != (h, w):
            depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR)
        
        # Normalize to [0, 1]
        depth = (depth - depth.min()) / (depth.max() - depth.min())
        
        # Scale to desired range
        depth_map = (depth * self.depth_scale).astype(np.float32)
        
        # Cache the result
        if self.use_cache:
            self.cache[img_hash] = (depth_map, (w, h))
        
        return depth_map, (w, h)
    
    def _bilinear_interpolate(self, depth_map, x, y):
        """
        Fast bilinear interpolation for sub-pixel accuracy.
        
        Args:
            depth_map: 2D depth array
            x, y: Coordinates (can be fractional)
            
        Returns:
            Interpolated depth value
        """
        h, w = depth_map.shape
        
        # Bounds checking
        x = np.clip(x, 0, w - 1.001)
        y = np.clip(y, 0, h - 1.001)
        
        # Get integer and fractional parts
        x0, y0 = int(np.floor(x)), int(np.floor(y))
        x1, y1 = min(x0 + 1, w - 1), min(y0 + 1, h - 1)
        fx, fy = x - x0, y - y0
        
        # Get corner values
        z00 = depth_map[y0, x0]
        z01 = depth_map[y0, x1]
        z10 = depth_map[y1, x0]
        z11 = depth_map[y1, x1]
        
        # Bilinear interpolation
        z = (z00 * (1 - fx) * (1 - fy) +
             z01 * fx * (1 - fy) +
             z10 * (1 - fx) * fy +
             z11 * fx * fy)
        
        return float(z)
    
    def getzpointfromxandy(self, annotatedimagepath, xcoordinate, ycoordinate):
        """
        Get Z (depth) coordinate for given X, Y pixel coordinates.
        
        Args:
            annotatedimagepath: Path to the image file
            xcoordinate: X pixel coordinate (can be fractional, e.g., 453.87)
            ycoordinate: Y pixel coordinate (can be fractional, e.g., 572.61)
            
        Returns:
            z_coordinate: Depth value at (x, y) with high accuracy
        """
        # Estimate depth map (uses cache if available)
        depth_map, img_size = self._estimate_depth_map(annotatedimagepath)
        
        # Validate coordinates
        w, h = img_size
        if not (0 <= xcoordinate < w and 0 <= ycoordinate < h):
            raise ValueError(
                f"Coordinates ({xcoordinate}, {ycoordinate}) out of bounds. "
                f"Image size: {w}x{h}"
            )
        
        # Get interpolated Z value for high accuracy
        z_coordinate = self._bilinear_interpolate(depth_map, xcoordinate, ycoordinate)
        
        return z_coordinate
    
    def get_multiple_z_points(self, annotatedimagepath, xy_coordinates):
        """
        Efficiently get Z values for multiple (x, y) coordinates.
        
        Args:
            annotatedimagepath: Path to the image file
            xy_coordinates: List of (x, y) tuples or 2D numpy array of shape (N, 2)
            
        Returns:
            z_coordinates: Numpy array of Z values
        """
        # Estimate depth map once
        depth_map, img_size = self._estimate_depth_map(annotatedimagepath)
        
        # Convert to numpy array if needed
        if isinstance(xy_coordinates, list):
            xy_coordinates = np.array(xy_coordinates)
        
        # Vectorized interpolation for multiple points
        z_values = np.array([
            self._bilinear_interpolate(depth_map, x, y)
            for x, y in xy_coordinates
        ])
        
        return z_values
    
    def get_full_point_cloud(self, annotatedimagepath):
        """
        Generate complete point cloud with RGB colors.
        
        Args:
            annotatedimagepath: Path to the image file
            
        Returns:
            points: Nx3 array of (x, y, z) coordinates
            colors: Nx3 array of RGB colors (0-1 range)
        """
        # Load image
        img = np.array(Image.open(annotatedimagepath).convert("RGB"))
        
        # Get depth map
        depth_map, img_size = self._estimate_depth_map(annotatedimagepath)
        
        w, h = img_size
        
        # Generate coordinate grids
        x_coords, y_coords = np.meshgrid(np.arange(w), np.arange(h), indexing='xy')
        
        # Stack coordinates
        points = np.column_stack([
            x_coords.flatten(),
            y_coords.flatten(),
            depth_map.flatten()
        ])
        
        # Get colors
        colors = img.reshape(-1, 3).astype(np.float32) / 255.0
        
        return points, colors
    
    def clear_cache(self):
        """Clear the depth map cache."""
        self.cache.clear()
        print("Cache cleared")
    
    def get_cache_info(self):
        """Get information about cached images."""
        return {
            "cached_images": len(self.cache),
            "cache_enabled": self.use_cache
        }


# ============================================================================
# Usage Examples
# ============================================================================

if __name__ == "__main__":
    # Initialize estimator (choose model size based on your needs)
    # - 'small': fastest, good accuracy
    # - 'base': balanced
    # - 'large': highest accuracy (recommended for production)
    
    estimator = DepthEstimator(model_size="large", depth_scale=1499.0, use_cache=True)
    
    # Example 1: Single point query
    image_path = "image.png"
    x, y = 453.87, 572.61
    
    print(f"\n--- Single Point Query ---")
    z = estimator.getzpointfromxandy(image_path, x, y)
    print(f"Coordinates: ({x}, {y}) -> Z = {z:.2f}")
    
    # Example 2: Multiple points (more efficient)
    print(f"\n--- Multiple Points Query ---")
    xy_points = [
        (100.0, 150.0),
        (453.87, 572.61),
        (200.5, 300.3),
        (500, 600)
    ]
    
    z_values = estimator.get_multiple_z_points(image_path, xy_points)
    for (x, y), z in zip(xy_points, z_values):
        print(f"  ({x}, {y}) -> Z = {z:.2f}")
    
    # Example 3: Repeated queries (uses cache - very fast)
    print(f"\n--- Cached Query (2nd call - should be instant) ---")
    z = estimator.getzpointfromxandy(image_path, 300, 400)
    print(f"Coordinates: (300, 400) -> Z = {z:.2f}")
    
    # Example 4: Get full point cloud
    print(f"\n--- Full Point Cloud Generation ---")
    points, colors = estimator.get_full_point_cloud(image_path)
    print(f"Generated {points.shape[0]} points")
    
    # Save point cloud
    np.savez_compressed("pointcloud_output.npz", points=points, colors=colors)
    print("Saved to pointcloud_output.npz")
    
    # Cache info
    print(f"\n--- Cache Info ---")
    print(estimator.get_cache_info())


# ============================================================================
# Integration Example for Your Class
# ============================================================================

class YourExistingClass:
    """Example of integrating DepthEstimator into your existing class."""
    
    def __init__(self):
        # Initialize depth estimator once (reuse for all calls)
        self.depth_estimator = DepthEstimator(
            model_size="large",      # High accuracy
            depth_scale=1499.0,
            use_cache=True           # Fast repeated queries
        )
    
    def getzpointfromxandy(self, annotatedimagepath, xcoordinate, ycoordinate):
        """
        Get Z coordinate for given X, Y pixel coordinates.
        High accuracy depth estimation using Depth Anything V2.
        """
        return self.depth_estimator.getzpointfromxandy(
            annotatedimagepath, 
            xcoordinate, 
            ycoordinate
        )
    
    def process_multiple_annotations(self, annotatedimagepath, annotations):
        """
        Process multiple annotations efficiently.
        
        Args:
            annotatedimagepath: Path to image
            annotations: List of (x, y) coordinates
            
        Returns:
            List of (x, y, z) coordinates
        """
        z_values = self.depth_estimator.get_multiple_z_points(
            annotatedimagepath, 
            annotations
        )
        
        # Combine with x, y coordinates
        xyz_coords = [
            (x, y, z) 
            for (x, y), z in zip(annotations, z_values)
        ]
        
        return xyz_coords