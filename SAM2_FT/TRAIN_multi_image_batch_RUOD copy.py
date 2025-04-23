# Train/Fine Tune SAM 2 on LabPics 1 dataset
# This mode use several images in a single batch
# Labpics can be downloaded from: https://zenodo.org/records/3697452/files/LabPicsV1.zip?download=1

import numpy as np
import torch
import cv2
import os
import json
from pycocotools import mask as mask_utils
from tqdm import tqdm
import datetime
from pathlib import Path
import gc
import matplotlib.pyplot as plt
from collections import defaultdict

from torch.onnx.symbolic_opset11 import hstack
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.cuda.amp import autocast, GradScaler

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# Enable memory efficient attention
torch.backends.cuda.enable_mem_efficient_sdp(True)
# Enable Flash attention if available
torch.backends.cuda.enable_flash_sdp(True)

# Read data
data_dir = "RUOD/"  # Path to RUOD dataset
train_ann_file = os.path.join(data_dir, "RUOD_ANN/instances_train.json")
val_ann_file = os.path.join(data_dir, "RUOD_ANN/instances_test.json")  # Using test set as validation
train_img_dir = os.path.join(data_dir, "RUOD_pic/train")
val_img_dir = os.path.join(data_dir, "RUOD_pic/test")

# Create output directory for saving models
output_dir = Path("outputs") / datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir.mkdir(parents=True, exist_ok=True)

def create_mask_from_bbox(height, width, bbox):
    """Create a binary mask from bbox coordinates [x, y, w, h]"""
    mask = np.zeros((height, width), dtype=np.uint8)
    x, y, w, h = map(int, bbox)
    mask[y:y+h, x:x+w] = 1
    return mask

# Load annotation files
def load_dataset(ann_file, img_dir):
    with open(ann_file, 'r') as f:
        annotations = json.load(f)
    
    # Create a mapping from image_id to annotations
    image_to_ann = {}
    for ann in annotations['annotations']:
        image_id = ann['image_id']
        if image_id not in image_to_ann:
            image_to_ann[image_id] = []
        image_to_ann[image_id].append(ann)
    
    # Create mapping from category_id to category name
    category_map = {cat['id']: cat['name'] for cat in annotations['categories']}
    
    # Create list of images with their metadata
    data = []
    for img in annotations['images']:
        if img['id'] in image_to_ann:  # only include images that have annotations
            data.append({
                "image": os.path.join(img_dir, img['file_name']),
                "height": img['height'],
                "width": img['width'],
                "id": img['id'],
                "annotations": image_to_ann[img['id']],
                "category_map": category_map
            })
    return data

train_data = load_dataset(train_ann_file, train_img_dir)
val_data = load_dataset(val_ann_file, val_img_dir)

def read_single(data):
    # select image
    ent = data[np.random.randint(len(data))]  # choose random entry
    Img = cv2.imread(ent["image"])
    if Img is None:
        print(f"Failed to load image: {ent['image']}")
        return read_single(data)  # retry if image loading fails
    
    Img = Img[...,::-1]  # BGR to RGB
    
    # Randomly select one annotation for this image
    ann = np.random.choice(ent["annotations"])
    
    # Create binary mask from bbox
    orig_h, orig_w = Img.shape[:2]
    mask = create_mask_from_bbox(orig_h, orig_w, ann['bbox'])
    
    # resize image while maintaining aspect ratio
    r = np.min([1024 / Img.shape[1], 1024 / Img.shape[0]])  # scaling factor
    new_size = (int(Img.shape[1] * r), int(Img.shape[0] * r))
    Img = cv2.resize(Img, new_size)
    mask = cv2.resize(mask, new_size, interpolation=cv2.INTER_NEAREST)
    
    # pad image and mask to 1024x1024
    padded_img = np.zeros((1024, 1024, 3), dtype=np.uint8)
    padded_mask = np.zeros((1024, 1024), dtype=np.uint8)
    padded_img[:Img.shape[0], :Img.shape[1]] = Img
    padded_mask[:mask.shape[0], :mask.shape[1]] = mask
    
    # Select a random point from the mask
    coords = np.argwhere(padded_mask > 0)
    if len(coords) == 0:  # if no points in mask, try another image
        return read_single(data)
    
    yx = coords[np.random.randint(len(coords))]
    return padded_img, padded_mask, [[yx[1], yx[0]]]

def read_batch(data, batch_size=4):
    limage = []
    lmask = []
    linput_point = []
    for i in range(batch_size):
        image, mask, input_point = read_single(data)
        limage.append(image)
        lmask.append(mask)
        linput_point.append(input_point)
    
    return limage, np.array(lmask), np.array(linput_point), np.ones([batch_size, 1])


# Load model

sam2_checkpoint = "checkpoints/sam2_hiera_small.pt" # path to model weight
model_cfg = "sam2_hiera_s.yaml" #  model config
sam2_model = build_sam2(model_cfg, sam2_checkpoint, device="cuda") # load model
predictor = SAM2ImagePredictor(sam2_model)

# Set training parameters

predictor.model.sam_mask_decoder.train(True) # enable training of mask decoder
predictor.model.sam_prompt_encoder.train(True) # enable training of prompt encoder
predictor.model.image_encoder.train(True) # enable training of image encoder: For this to work you need to scan the code for "no_grad" and remove them all
optimizer = torch.optim.AdamW(params=predictor.model.parameters(), lr=1e-5, weight_decay=4e-5)
scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True)
scaler = torch.cuda.amp.GradScaler() # mixed precision

# Training configuration
num_epochs = 16
steps_per_epoch = 1000
best_val_iou = 0
patience = 15
patience_counter = 0

# Initialize metrics storage
metrics = defaultdict(list)

def validate(data, num_steps=100):
    predictor.model.eval()
    total_iou = 0
    total_loss = 0
    val_batch_size = 2  # Reduced batch size for validation
    
    try:
        with torch.no_grad(), autocast(enabled=True):  # Enable mixed precision for validation
            for step in range(num_steps):
                # Clear cache
                if step % 10 == 0:
                    torch.cuda.empty_cache()
                    gc.collect()
                
                image, mask, input_point, input_label = read_batch(data, batch_size=val_batch_size)
                if mask.shape[0] == 0: 
                    continue
                
                predictor.set_image_batch(image)
                mask_input, unnorm_coords, labels, unnorm_box = predictor._prep_prompts(
                    input_point, input_label, box=None, mask_logits=None, normalize_coords=True
                )
                
                # Convert inputs to half precision
                sparse_embeddings, dense_embeddings = predictor.model.sam_prompt_encoder(
                    points=(unnorm_coords.half(), labels.half()), 
                    boxes=None, 
                    masks=None,
                )
                
                high_res_features = [feat_level[-1].unsqueeze(0).half() for feat_level in predictor._features["high_res_feats"]]
                
                try:
                    low_res_masks, prd_scores, _, _ = predictor.model.sam_mask_decoder(
                        image_embeddings=predictor._features["image_embed"].half(),
                        image_pe=predictor.model.sam_prompt_encoder.get_dense_pe().half(),
                        sparse_prompt_embeddings=sparse_embeddings,
                        dense_prompt_embeddings=dense_embeddings,
                        multimask_output=True,
                        repeat_image=False,
                        high_res_features=high_res_features,
                    )
                except RuntimeError as e:
                    print(f"Error in validation step {step}: {e}")
                    continue
                
                prd_masks = predictor._transforms.postprocess_masks(low_res_masks, predictor._orig_hw[-1])
                
                gt_mask = torch.tensor(mask.astype(np.float32)).cuda().half()
                prd_mask = torch.sigmoid(prd_masks[:, 0])
                
                # Calculate metrics
                inter = (gt_mask * (prd_mask > 0.5)).sum(1).sum(1)
                union = gt_mask.sum(1).sum(1) + (prd_mask > 0.5).sum(1).sum(1) - inter
                iou = (inter / union).mean()
                total_iou += iou.item()
                
                # Calculate loss
                seg_loss = (-gt_mask * torch.log(prd_mask + 0.00001) - (1 - gt_mask) * torch.log((1 - prd_mask) + 0.00001)).mean()
                score_loss = torch.abs(prd_scores[:, 0] - iou).mean()
                loss = seg_loss + score_loss * 0.05
                total_loss += loss.item()
                
                # Store step metrics
                metrics['step_loss'].append(loss.item())
                metrics['step_iou'].append(iou.item())
                
                # Clear some memory
                del low_res_masks, prd_masks, gt_mask, prd_mask
                torch.cuda.empty_cache()
    
    except Exception as e:
        print(f"Validation error: {e}")
        return 0.0, 0.0
    
    finally:
        predictor.model.train()
        torch.cuda.empty_cache()
        gc.collect()
    
    return total_iou / num_steps, total_loss / num_steps

# Training loop
for epoch in range(num_epochs):
    torch.cuda.empty_cache()
    gc.collect()
    
    epoch_loss = 0
    epoch_iou = 0
    
    # Training
    progress_bar = tqdm(range(steps_per_epoch), desc=f'Epoch {epoch+1}/{num_epochs}')
    for step in progress_bar:
        with autocast():  # Enable mixed precision for training
            image, mask, input_point, input_label = read_batch(train_data, batch_size=4)
            if mask.shape[0] == 0: 
                continue
            
            predictor.set_image_batch(image)
            mask_input, unnorm_coords, labels, unnorm_box = predictor._prep_prompts(
                input_point, input_label, box=None, mask_logits=None, normalize_coords=True
            )
            
            # Convert inputs to half precision
            sparse_embeddings, dense_embeddings = predictor.model.sam_prompt_encoder(
                points=(unnorm_coords.half(), labels.half()),
                boxes=None,
                masks=None,
            )
            
            high_res_features = [feat_level[-1].unsqueeze(0).half() for feat_level in predictor._features["high_res_feats"]]
            low_res_masks, prd_scores, _, _ = predictor.model.sam_mask_decoder(
                image_embeddings=predictor._features["image_embed"].half(),
                image_pe=predictor.model.sam_prompt_encoder.get_dense_pe().half(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=True,
                repeat_image=False,
                high_res_features=high_res_features,
            )
            prd_masks = predictor._transforms.postprocess_masks(low_res_masks, predictor._orig_hw[-1])
            
            gt_mask = torch.tensor(mask.astype(np.float32)).cuda().half()
            prd_mask = torch.sigmoid(prd_masks[:, 0])
            
            # Calculate loss
            seg_loss = (-gt_mask * torch.log(prd_mask + 0.00001) - (1 - gt_mask) * torch.log((1 - prd_mask) + 0.00001)).mean()
            inter = (gt_mask * (prd_mask > 0.5)).sum(1).sum(1)
            union = gt_mask.sum(1).sum(1) + (prd_mask > 0.5).sum(1).sum(1) - inter
            iou = (inter / union).mean()
            score_loss = torch.abs(prd_scores[:, 0] - iou).mean()
            loss = seg_loss + score_loss * 0.05
            
            epoch_loss += loss.item()
            epoch_iou += iou.item()
        
        # Backpropagation
        predictor.model.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # Clear some memory
        del low_res_masks, prd_masks, gt_mask, prd_mask
        if step % 50 == 0:
            torch.cuda.empty_cache()
            gc.collect()
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': f'{epoch_loss/(step+1):.4f}',
            'iou': f'{epoch_iou/(step+1):.4f}'
        })
    
    # Calculate epoch metrics
    train_loss = epoch_loss / steps_per_epoch
    train_iou = epoch_iou / steps_per_epoch
    
    # Validation
    val_iou, val_loss = validate(val_data)
    
    # Store epoch metrics
    metrics['epoch'].append(epoch + 1)
    metrics['train_loss'].append(train_loss)
    metrics['train_iou'].append(train_iou)
    metrics['val_loss'].append(val_loss)
    metrics['val_iou'].append(val_iou)
    metrics['learning_rate'].append(optimizer.param_groups[0]['lr'])
    
    # Update learning rate
    scheduler.step(val_iou)
    
    # Save checkpoint
    checkpoint = {
        'epoch': epoch + 1,
        'model_state_dict': predictor.model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'train_loss': train_loss,
        'train_iou': train_iou,
        'val_loss': val_loss,
        'val_iou': val_iou,
        'metrics': metrics,  # Save all metrics in checkpoint
    }
    
    # Save latest checkpoint
    torch.save(checkpoint, output_dir / f'checkpoint_latest.pth')
    
    # Save best model
    if val_iou > best_val_iou:
        best_val_iou = val_iou
        torch.save(checkpoint, output_dir / f'checkpoint_best.pth')
        patience_counter = 0
    else:
        patience_counter += 1
    
    # Print epoch results
    print(f'\nEpoch {epoch+1}/{num_epochs}:')
    print(f'Train Loss: {train_loss:.4f}, Train IOU: {train_iou:.4f}')
    print(f'Val Loss: {val_loss:.4f}, Val IOU: {val_iou:.4f}')
    print(f'Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')
    
    # Plot and save metrics every 5 epochs or at the end
    if (epoch + 1) % 5 == 0 or epoch == num_epochs - 1:
        # Create figures directory
        figures_dir = output_dir / 'figures'
        figures_dir.mkdir(exist_ok=True)
        
        # Set style for better visualization
        plt.style.use('bmh')  # Using built-in style instead of seaborn
        
        # Plot Loss
        plt.figure(figsize=(10, 6))
        plt.plot(metrics['epoch'], metrics['train_loss'], label='Train Loss', marker='o', linestyle='-', linewidth=2)
        plt.plot(metrics['epoch'], metrics['val_loss'], label='Val Loss', marker='s', linestyle='--', linewidth=2)
        plt.title('Training and Validation Loss Over Time', fontsize=12, pad=15)
        plt.xlabel('Epoch', fontsize=10)
        plt.ylabel('Loss', fontsize=10)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(figures_dir / 'loss_plot.png', bbox_inches='tight', dpi=300)
        plt.close()
        
        # Plot IOU
        plt.figure(figsize=(10, 6))
        plt.plot(metrics['epoch'], metrics['train_iou'], label='Train IOU', marker='o', linestyle='-', linewidth=2)
        plt.plot(metrics['epoch'], metrics['val_iou'], label='Val IOU', marker='s', linestyle='--', linewidth=2)
        plt.title('Training and Validation IOU Over Time', fontsize=12, pad=15)
        plt.xlabel('Epoch', fontsize=10)
        plt.ylabel('IOU', fontsize=10)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(figures_dir / 'iou_plot.png', bbox_inches='tight', dpi=300)
        plt.close()
        
        # Plot Learning Rate
        plt.figure(figsize=(10, 6))
        plt.plot(metrics['epoch'], metrics['learning_rate'], marker='o', linestyle='-', linewidth=2, color='#2ecc71')
        plt.title('Learning Rate Over Time', fontsize=12, pad=15)
        plt.xlabel('Epoch', fontsize=10)
        plt.ylabel('Learning Rate', fontsize=10)
        plt.yscale('log')  # Use log scale for learning rate
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(figures_dir / 'lr_plot.png', bbox_inches='tight', dpi=300)
        plt.close()
        
        # Plot Step-wise metrics for the last epoch
        plt.figure(figsize=(15, 5))
        
        # Step Loss
        plt.subplot(1, 2, 1)
        steps_in_epoch = len(metrics['step_loss']) // (epoch + 1)
        last_epoch_losses = metrics['step_loss'][-steps_in_epoch:]
        plt.plot(range(len(last_epoch_losses)), last_epoch_losses, linewidth=1, color='#3498db')
        plt.title(f'Step-wise Loss (Epoch {epoch + 1})', fontsize=12, pad=15)
        plt.xlabel('Step', fontsize=10)
        plt.ylabel('Loss', fontsize=10)
        plt.grid(True, alpha=0.3)
        
        # Step IOU
        plt.subplot(1, 2, 2)
        last_epoch_ious = metrics['step_iou'][-steps_in_epoch:]
        plt.plot(range(len(last_epoch_ious)), last_epoch_ious, linewidth=1, color='#e74c3c')
        plt.title(f'Step-wise IOU (Epoch {epoch + 1})', fontsize=12, pad=15)
        plt.xlabel('Step', fontsize=10)
        plt.ylabel('IOU', fontsize=10)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(figures_dir / f'step_metrics_epoch_{epoch+1}.png', bbox_inches='tight', dpi=300)
        plt.close()
        
        try:
            # Save metrics to CSV
            import pandas as pd
            metrics_df = pd.DataFrame({
                'epoch': metrics['epoch'],
                'train_loss': metrics['train_loss'],
                'val_loss': metrics['val_loss'],
                'train_iou': metrics['train_iou'],
                'val_iou': metrics['val_iou'],
                'learning_rate': metrics['learning_rate']
            })
            metrics_df.to_csv(output_dir / 'training_metrics.csv', index=False)
        except ImportError:
            print("Warning: pandas not installed, skipping CSV export")
            
        # Clear matplotlib memory
        plt.clf()
        plt.close('all')
    
    # Early stopping
    if patience_counter >= patience:
        print(f'\nEarly stopping triggered after {epoch+1} epochs')
        break

print('Training completed!')

# Final plots
print('Saving final metric plots...')
# Create final summary plots (they were already created in the last epoch)
print(f'All plots have been saved to {figures_dir}')
print(f'Training metrics have been saved to {output_dir}/training_metrics.csv')
