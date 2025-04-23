# Segment image region using fine-tuned SAM2 model on UFO120 dataset
import numpy as np
import torch
import cv2
import os
from pathlib import Path
import matplotlib.pyplot as plt
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

def load_best_model(output_dir):
    """Load the best model from training output directory"""
    checkpoint_path = os.path.join(output_dir, 'checkpoint_best.pth')
    print(f"Loading checkpoint from: {checkpoint_path}")
    
    # Add defaultdict to safe globals for loading
    torch.serialization.add_safe_globals(['defaultdict'])
    
    try:
        # First try loading with weights_only=False
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        print("Successfully loaded full checkpoint")
    except Exception as e:
        print(f"Full checkpoint loading failed, trying weights only: {e}")
        try:
            # Fallback to weights_only=True
            checkpoint = torch.load(checkpoint_path, weights_only=True)
            print("Successfully loaded weights only")
        except Exception as e:
            raise Exception(f"Failed to load checkpoint: {e}")
    
    return checkpoint

# use bfloat16 for the entire script (memory efficient)
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

def read_image(image_path, mask_path=None):
    """Read and resize image and optional mask"""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image from {image_path}")
    img = img[...,::-1]  # BGR to RGB

    # Resize image to maximum size of 1024
    r = np.min([1024 / img.shape[1], 1024 / img.shape[0]])
    img = cv2.resize(img, (int(img.shape[1] * r), int(img.shape[0] * r)))
    
    if mask_path is not None and os.path.exists(mask_path):
        mask = cv2.imread(mask_path, 0)  # Read as grayscale
        mask = (mask > 0).astype(np.uint8)  # Binary mask
        mask = cv2.resize(mask, (int(mask.shape[1] * r), int(mask.shape[0] * r)), interpolation=cv2.INTER_NEAREST)
        return img, mask
    return img, None

def get_points(image, mask, num_points):
    """Sample points inside the input mask or in a grid pattern"""
    points = []
    if mask is None:
        # If no mask provided, sample points in a grid
        h, w = image.shape[:2]
        step = int(np.sqrt(h * w / num_points))
        for y in range(step//2, h, step):
            for x in range(step//2, w, step):
                points.append([[x, y]])
                if len(points) >= num_points:
                    break
            if len(points) >= num_points:
                break
    else:
        # Sample points from mask
        coords = np.argwhere(mask > 0)
        if len(coords) == 0:
            raise ValueError("Mask is empty")
        for i in range(min(num_points, len(coords))):
            idx = np.random.randint(len(coords))
            yx = coords[idx]
            points.append([[yx[1], yx[0]]])
    return np.array(points)

def visualize_results(image, masks, scores, save_dir):
    """Visualize and save segmentation results"""
    # Create save directory if it doesn't exist
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving results to: {save_dir}")

    # Convert tensors to numpy if needed
    if isinstance(masks, torch.Tensor):
        masks = masks.cpu().numpy()
    if isinstance(scores, torch.Tensor):
        scores = scores.cpu().numpy()

    # Print shape information for debugging
    print(f"Masks shape: {masks.shape}")
    print(f"Scores shape: {scores.shape}")

    # Ensure masks are in the correct format
    if len(masks.shape) == 4:  # If shape is (N, 3, H, W)
        # Take the mask with highest score for each point
        best_mask_indices = scores.argmax(axis=1)
        masks = np.stack([masks[i, idx] for i, idx in enumerate(best_mask_indices)])
        scores = np.array([scores[i, idx] for i, idx in enumerate(best_mask_indices)])
    
    print(f"Processing {len(masks)} masks")
    
    # Create segmentation map
    seg_map = np.zeros(masks[0].shape, dtype=np.uint8)
    occupancy_mask = np.zeros_like(masks[0], dtype=bool)
    
    # Generate random colors for visualization
    colors = np.random.randint(0, 255, (len(masks), 3))
    
    # Create RGB visualization
    rgb_image = np.zeros((*seg_map.shape, 3), dtype=np.uint8)
    
    # Sort masks by score
    sorted_idx = np.argsort(scores)[::-1]
    masks = masks[sorted_idx]
    scores = scores[sorted_idx]
    
    num_valid_masks = 0
    for i, (mask, score) in enumerate(zip(masks, scores)):
        if isinstance(mask, torch.Tensor):
            mask = mask.cpu().numpy()
        mask = mask.astype(bool)
        
        # Check overlap with existing masks
        overlap_ratio = (mask & occupancy_mask).sum() / (mask.sum() + 1e-6)
        if overlap_ratio > 0.15:
            continue
            
        mask = mask & ~occupancy_mask  # Remove overlapping regions
        if mask.sum() > 0:  # Only add mask if there are pixels remaining
            seg_map[mask] = num_valid_masks + 1
            rgb_image[mask] = colors[num_valid_masks]
            occupancy_mask |= mask
            num_valid_masks += 1
            if isinstance(score, np.ndarray):
                score = float(score.max())  # Convert array to scalar if needed
            print(f"Segment {num_valid_masks}: Score = {score:.4f}")

    print(f"\nGenerated {num_valid_masks} valid segments")

    # Save results
    output_files = {
        "segmentation_map.png": seg_map,
        "segmentation_colored.png": rgb_image,
        "segmentation_overlay.png": (rgb_image * 0.5 + image * 0.5).astype(np.uint8),
    }
    
    for filename, img in output_files.items():
        filepath = save_dir / filename
        cv2.imwrite(str(filepath), img if len(img.shape) == 2 else img[..., ::-1])
        print(f"Saved {filepath}")

    # Create and save visualization plot
    plt.figure(figsize=(15, 5))
    
    plt.subplot(131)
    plt.imshow(image)
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(132)
    plt.imshow(rgb_image)
    plt.title(f'Segmentation\n({num_valid_masks} segments)')
    plt.axis('off')
    
    plt.subplot(133)
    plt.imshow((rgb_image * 0.5 + image * 0.5).astype(np.uint8))
    plt.title('Overlay')
    plt.axis('off')
    
    plt.tight_layout()
    viz_path = save_dir / "visualization.png"
    plt.savefig(str(viz_path), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved visualization to {viz_path}")

    return seg_map, rgb_image

def main():
    # Configuration
    image_path = "UFO120/train/lrd/set_f100.jpg"  # path to test image
    mask_path = "UFO120/train/mask/set_f100.jpg"  # Optional: path to mask for guided testing (can be None)
    num_samples = 30  # number of points to sample
    output_dir = "outputs"  # directory containing training outputs
    results_dir = "test_results"  # directory to save test results
    
    print("\nConfiguration:")
    print(f"Image path: {image_path}")
    print(f"Mask path: {mask_path}")
    print(f"Number of sample points: {num_samples}")
    print(f"Results directory: {results_dir}")
    
    # Load model
    sam2_checkpoint = "checkpoints/sam2_hiera_small.pt"
    model_cfg = "sam2_hiera_s.yaml"
    print(f"\nLoading base model from: {sam2_checkpoint}")
    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device="cuda")
    predictor = SAM2ImagePredictor(sam2_model)
    
    # Load the best model from training
    latest_output = max(Path(output_dir).glob('*'), key=os.path.getctime)
    print(f"\nFound latest training output: {latest_output}")
    checkpoint = load_best_model(latest_output)
    predictor.model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded model from epoch {checkpoint['epoch']} with validation IOU: {checkpoint['val_iou']:.4f}")
    
    # Read image and get points
    print(f"\nProcessing image: {image_path}")
    image, mask = read_image(image_path, mask_path)
    input_points = get_points(image, mask, num_samples)
    print(f"Generated {len(input_points)} sample points")
    
    # Predict masks
    print("\nGenerating predictions...")
    with torch.no_grad():
        predictor.set_image(image)
        masks, scores, logits = predictor.predict(
            point_coords=input_points,
            point_labels=np.ones([input_points.shape[0], 1])
        )
    print(f"Generated {len(masks)} masks")
    
    # Convert masks to numpy if they're torch tensors
    if isinstance(masks, torch.Tensor):
        masks = masks.cpu().numpy()
    if isinstance(scores, torch.Tensor):
        scores = scores.cpu().numpy()
    
    # Visualize and save results
    test_save_dir = Path(results_dir) / Path(image_path).stem
    seg_map, rgb_image = visualize_results(image, masks, scores, test_save_dir)
    
    # Display results (optional)
    cv2.imshow("Original", image[..., ::-1])  # Convert back to BGR for display
    cv2.imshow("Segmentation", rgb_image)
    cv2.imshow("Overlay", (rgb_image * 0.5 + image * 0.5).astype(np.uint8))
    print("\nPress any key to close the display windows...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()