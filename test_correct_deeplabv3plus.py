import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os
from tqdm import tqdm
import segmentation_models_pytorch as smp

class DeepLabV3PlusMobileNet(nn.Module):
    def __init__(self, num_classes=19):
        super().__init__()
        # Use the same architecture as the pretrained model
        self.model = smp.DeepLabV3Plus(
            encoder_name="mobilenet_v2",
            encoder_weights=None,  # We'll load our pretrained weights
            classes=num_classes,
            activation=None
        )
    
    def forward(self, x):
        return self.model(x)

class SegmentationDataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        
        self.images = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])
        self.labels = sorted([f for f in os.listdir(label_dir) if f.endswith('.png')])
        
        # Ensure image and label files match
        assert len(self.images) == len(self.labels), f"Mismatch: {len(self.images)} images vs {len(self.labels)} labels"
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # Load image
        img_path = os.path.join(self.image_dir, self.images[idx])
        image = Image.open(img_path).convert('RGB')
        
        # Load label
        label_path = os.path.join(self.label_dir, self.labels[idx])
        label = Image.open(label_path)
        
        # Convert to numpy arrays
        image = np.array(image)
        label = np.array(label)
        
        # Apply transforms if provided
        if self.transform:
            transformed = self.transform(image=image, mask=label)
            image = transformed['image']
            label = transformed['mask']
        else:
            # Default transforms for testing
            image = transforms.ToTensor()(image)
            # Normalize with ImageNet statistics (commonly used for pretrained models)
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                           std=[0.229, 0.224, 0.225])
            image = normalize(image)
            label = torch.tensor(label, dtype=torch.long)
        
        return image, label

def calculate_miou(pred, target, num_classes):
    """Calculate mean Intersection over Union (mIoU)"""
    ious = []
    pred = pred.view(-1)
    target = target.view(-1)
    
    for cls in range(num_classes):
        pred_inds = pred == cls
        target_inds = target == cls
        intersection = (pred_inds[target_inds]).long().sum().data.cpu().item()
        union = pred_inds.long().sum().data.cpu().item() + target_inds.long().sum().data.cpu().item() - intersection
        
        if union == 0:
            ious.append(float('nan'))  # If there is no ground truth, do not include in evaluation
        else:
            ious.append(float(intersection) / float(max(union, 1)))
    
    return ious

def test_model():
    # Load class names
    with open('seg_data/names.txt', 'r') as f:
        class_names = [line.strip() for line in f.readlines()]
    
    num_classes = len(class_names)
    print(f"Number of classes: {num_classes}")
    print(f"Classes: {class_names}")
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model with correct architecture
    model = DeepLabV3PlusMobileNet(num_classes=num_classes)
    
    # Load pretrained weights
    print("Loading pretrained model...")
    model_path = "best_deeplabv3plus_mobilenet_cityscapes_os16.pth"
    
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        print("Checkpoint keys:", list(checkpoint.keys()))
        
        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            print("Using 'model_state_dict' from checkpoint")
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            print("Using 'state_dict' from checkpoint")
        else:
            state_dict = checkpoint
            print("Using entire checkpoint as state_dict")
        
        # Try to load weights
        try:
            model.load_state_dict(state_dict, strict=True)
            print("✓ Loaded pretrained weights successfully (strict=True)")
        except RuntimeError as e:
            print(f"Strict loading failed: {e}")
            print("Trying to load with strict=False...")
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            print(f"Missing keys: {len(missing_keys)}")
            print(f"Unexpected keys: {len(unexpected_keys)}")
            if missing_keys:
                print("First 5 missing keys:", missing_keys[:5])
            if unexpected_keys:
                print("First 5 unexpected keys:", unexpected_keys[:5])
        
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    model.to(device)
    model.eval()
    
    # Create dataset and dataloader
    test_dataset = SegmentationDataset(
        image_dir='seg_data/testing/image',
        label_dir='seg_data/testing/label'
    )
    
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)
    
    print(f"Test dataset size: {len(test_dataset)}")
    
    # Test the model
    all_ious = []
    class_ious = [[] for _ in range(num_classes)]
    
    print("Testing model...")
    with torch.no_grad():
        for i, (images, labels) in enumerate(tqdm(test_loader, desc="Testing")):
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(images)
            
            # Get predictions
            if isinstance(outputs, dict):
                # Handle case where model returns dictionary
                predictions = outputs['out'] if 'out' in outputs else list(outputs.values())[0]
            else:
                predictions = outputs
            
            predictions = torch.argmax(predictions, dim=1)
            
            # Calculate IoU for this batch
            batch_ious = calculate_miou(predictions, labels, num_classes)
            
            # Store per-class IoUs
            for cls_idx, iou in enumerate(batch_ious):
                if not np.isnan(iou):
                    class_ious[cls_idx].append(iou)
            
            all_ious.extend([iou for iou in batch_ious if not np.isnan(iou)])
            
            # Print progress for first few samples
            if i < 5:
                valid_ious = [iou for iou in batch_ious if not np.isnan(iou)]
                print(f"Sample {i+1}: mIoU = {np.mean(valid_ious):.4f}")
    
    # Calculate final metrics
    overall_miou = np.mean(all_ious)
    
    print("\n" + "="*50)
    print("FINAL RESULTS")
    print("="*50)
    print(f"Overall mIoU: {overall_miou:.4f}")
    
    # Per-class IoU
    print("\nPer-class IoU:")
    print("-" * 40)
    for i, class_name in enumerate(class_names):
        if class_ious[i]:
            class_miou = np.mean(class_ious[i])
            print(f"{class_name:15}: {class_miou:.4f} ({len(class_ious[i])} samples)")
        else:
            print(f"{class_name:15}: N/A (no samples)")
    
    # Save results
    with open('test_results_deeplabv3plus_correct.txt', 'w') as f:
        f.write(f"DeepLabV3+ MobileNet Test Results\n")
        f.write(f"================================\n\n")
        f.write(f"Overall mIoU: {overall_miou:.4f}\n\n")
        f.write(f"Per-class IoU:\n")
        for i, class_name in enumerate(class_names):
            if class_ious[i]:
                class_miou = np.mean(class_ious[i])
                f.write(f"{class_name}: {class_miou:.4f}\n")
            else:
                f.write(f"{class_name}: N/A\n")
    
    print(f"\nResults saved to 'test_results_deeplabv3plus_correct.txt'")

if __name__ == "__main__":
    test_model()
