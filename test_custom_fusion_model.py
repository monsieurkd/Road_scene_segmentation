#!/usr/bin/env python3
"""
Test script for custom UNetDeepLabFusion model for road scene segmentation
Loads the actual trained model architecture and calculates mIoU on test dataset
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import os
import glob
from pathlib import Path

# Import the required components
try:
    from torchvision.models import efficientnet_b0
    print("✅ EfficientNet-B0 available in torchvision")
except ImportError:
    print("❌ EfficientNet not available in torchvision")
    efficientnet_b0 = None

class EfficientNetEncoder(nn.Module):
    """EfficientNet-B0 encoder with skip connections for U-Net style architecture"""
    def __init__(self, pretrained=True):
        super(EfficientNetEncoder, self).__init__()
        
        # Load EfficientNet-B0
        if efficientnet_b0 is not None:
            self.backbone = efficientnet_b0(pretrained=pretrained)
            self.backbone_type = "torchvision"
        else:
            raise ImportError("EfficientNet-B0 not available")
        
        # Extract feature layers for skip connections
        self.features = self.backbone.features
        # EfficientNet-B0 feature channels: [16, 24, 40, 80, 112, 192, 320, 1280]
        self.skip_channels = [16, 24, 40, 80, 112, 192, 320]
        self.final_channels = 1280
    
    def forward(self, x):
        skip_features = []
        
        # Extract features at different scales
        for i, layer in enumerate(self.features):
            x = layer(x)
            # Collect skip connection features at specific indices
            if i in [1, 2, 3, 4, 5, 6, 7]:  # Adjust indices based on EfficientNet-B0 structure
                skip_features.append(x)
        
        return x, skip_features

class ASPP(nn.Module):
    """Enhanced Atrous Spatial Pyramid Pooling module"""
    def __init__(self, in_channels, out_channels=256, rates=[6, 12, 18, 24]):
        super(ASPP, self).__init__()

        # 1x1 convolution
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        # Atrous convolutions with different dilation rates
        self.atrous_convs = nn.ModuleList()
        for rate in rates:
            self.atrous_convs.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=rate, 
                         dilation=rate, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ))

        # Global average pooling branch
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        # Final projection layer
        total_channels = out_channels * (len(rates) + 2)  # +2 for 1x1 conv and global pooling
        self.project = nn.Sequential(
            nn.Conv2d(total_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )

    def forward(self, x):
        size = x.shape[-2:]
        
        # Apply different ASPP branches
        features = [self.conv1(x)]
        
        # Add atrous convolutions
        for atrous_conv in self.atrous_convs:
            features.append(atrous_conv(x))
        
        # Add global average pooling
        global_feat = F.interpolate(self.global_avg_pool(x), size=size, 
                                  mode='bilinear', align_corners=False)
        features.append(global_feat)

        # Concatenate all branches
        x = torch.cat(features, dim=1)
        x = self.project(x)

        return x

class FusionBlock(nn.Module):
    """Feature fusion block for combining skip connections with upsampled features"""
    def __init__(self, high_channels, skip_channels, out_channels):
        super(FusionBlock, self).__init__()
        
        # Process skip connection features
        self.skip_conv = nn.Sequential(
            nn.Conv2d(skip_channels, out_channels//2, 1, bias=False),
            nn.BatchNorm2d(out_channels//2),
            nn.ReLU(inplace=True)
        )
        
        # Process high-level features
        self.high_conv = nn.Sequential(
            nn.Conv2d(high_channels, out_channels//2, 1, bias=False),
            nn.BatchNorm2d(out_channels//2),
            nn.ReLU(inplace=True)
        )
        
        # Fusion convolution
        self.fusion = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, high_feat, skip_feat):
        # Upsample high-level features to match skip features
        high_feat = F.interpolate(high_feat, size=skip_feat.shape[-2:], 
                                mode='bilinear', align_corners=False)
        
        # Process features
        skip_processed = self.skip_conv(skip_feat)
        high_processed = self.high_conv(high_feat)
        
        # Concatenate and fuse
        fused = torch.cat([skip_processed, high_processed], dim=1)
        return self.fusion(fused)

class UNetDeepLabFusion(nn.Module):
    """Fusion model combining U-Net decoder with DeepLabV3+ ASPP and EfficientNet-B0 encoder"""
    def __init__(self, num_classes=19, pretrained=True):
        super(UNetDeepLabFusion, self).__init__()
        
        self.num_classes = num_classes
        
        # EfficientNet-B0 encoder
        self.encoder = EfficientNetEncoder(pretrained=pretrained)
        
        # ASPP module for high-level feature processing
        self.aspp = ASPP(self.encoder.final_channels, 256)
        
        # Decoder fusion blocks - progressive upsampling with skip connections
        # EfficientNet-B0 skip channels: [16, 24, 40, 80, 112, 192, 320]
        self.decoder_channels = [256, 192, 128, 96, 64, 48, 32]
        
        self.fusion_blocks = nn.ModuleList([
            FusionBlock(256, 320, self.decoder_channels[0]),  # Highest skip connection
            FusionBlock(self.decoder_channels[0], 192, self.decoder_channels[1]),
            FusionBlock(self.decoder_channels[1], 112, self.decoder_channels[2]),
            FusionBlock(self.decoder_channels[2], 80, self.decoder_channels[3]),
            FusionBlock(self.decoder_channels[3], 40, self.decoder_channels[4]),
            FusionBlock(self.decoder_channels[4], 24, self.decoder_channels[5]),
            FusionBlock(self.decoder_channels[5], 16, self.decoder_channels[6])
        ])
        
        # Final upsampling and classification
        self.final_upsample = nn.Sequential(
            nn.ConvTranspose2d(self.decoder_channels[6], 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        
        # Final classifier
        self.classifier = nn.Conv2d(16, num_classes, 1)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        input_size = x.shape[-2:]
        
        # Extract features from encoder
        features, skip_features = self.encoder(x)
        
        # Process through ASPP
        x = self.aspp(features)
        
        # Progressive decoding with skip connections
        # Reverse skip features to match decoder order (highest resolution first)
        skip_features = skip_features[::-1]
        
        for i, fusion_block in enumerate(self.fusion_blocks):
            if i < len(skip_features):
                x = fusion_block(x, skip_features[i])
            else:
                # If we run out of skip features, just upsample
                x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        
        # Final upsampling and classification
        x = self.final_upsample(x)
        x = self.classifier(x)
        
        # Ensure output matches input size
        if x.shape[-2:] != input_size:
            x = F.interpolate(x, size=input_size, mode='bilinear', align_corners=False)
        
        return x
    
    def get_parameter_groups(self, lr_encoder=1e-4, lr_decoder=1e-3):
        """Get parameter groups for differential learning rates"""
        encoder_params = list(self.encoder.parameters())
        decoder_params = list(self.aspp.parameters()) + \
                        list(self.fusion_blocks.parameters()) + \
                        list(self.final_upsample.parameters()) + \
                        list(self.classifier.parameters())
        
        return [
            {'params': encoder_params, 'lr': lr_encoder},
            {'params': decoder_params, 'lr': lr_decoder}
        ]

class RoadDataset(Dataset):
    """Dataset for road scene segmentation"""
    
    def __init__(self, image_dir, label_dir, transform=None, target_transform=None):
        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir)
        self.transform = transform
        self.target_transform = target_transform
        
        # Get all image files
        self.image_files = sorted(glob.glob(str(self.image_dir / "*.png")))
        self.label_files = sorted(glob.glob(str(self.label_dir / "*.png")))
        
        print(f"Found {len(self.image_files)} images and {len(self.label_files)} labels")
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Load image
        image_path = self.image_files[idx]
        image = Image.open(image_path).convert('RGB')
        
        # Load label
        label_path = self.label_files[idx]
        label = Image.open(label_path)
        label = np.array(label)
        
        # Convert to tensor
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        
        return image, label, os.path.basename(image_path)

def calculate_miou(pred, target, num_classes):
    """Calculate Mean Intersection over Union"""
    ious = []
    pred = pred.flatten()
    target = target.flatten()
    
    for class_id in range(num_classes):
        pred_mask = pred == class_id
        target_mask = target == class_id
        
        if target_mask.sum() == 0:  # No ground truth for this class
            if pred_mask.sum() == 0:  # No prediction either
                ious.append(float('nan'))  # Skip this class
            else:
                ious.append(0.0)  # False positive
        else:
            intersection = (pred_mask & target_mask).sum()
            union = (pred_mask | target_mask).sum()
            iou = float(intersection) / float(union)
            ious.append(iou)
    
    # Calculate mean IoU, ignoring NaN values
    valid_ious = [iou for iou in ious if not np.isnan(iou)]
    return np.mean(valid_ious), ious

def load_pretrained_model(model_path, num_classes=19):
    """Load the custom pretrained UNetDeepLabFusion model"""
    print(f"Loading pretrained model from: {model_path}")
    
    # Create model
    model = UNetDeepLabFusion(num_classes=num_classes, pretrained=False)
    
    try:
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        print("Checkpoint loaded successfully")
        
        # Extract model state dict
        if 'model_state' in checkpoint:
            state_dict = checkpoint['model_state']
            print("Using 'model_state' from checkpoint")
        else:
            state_dict = checkpoint
            print("Using checkpoint directly as state dict")
        
        # Load state dict
        model.load_state_dict(state_dict)
        print("Model state dict loaded successfully")
        
        if 'best_score' in checkpoint:
            print(f"Model's best training score: {checkpoint['best_score']}")
        if 'cur_itrs' in checkpoint:
            print(f"Training iterations: {checkpoint['cur_itrs']}")
        
        return model
        
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Using randomly initialized model...")
        return model

def main():
    """Main testing function"""
    print("="*60)
    print("CUSTOM UNET-DEEPLABV3+ FUSION MODEL TESTING")
    print("="*60)
    
    # Setup paths
    base_dir = Path("/home/kieuduc/Desktop/Road_scene_segmentation")
    model_path = base_dir / "best_deeplabv3plus_mobilenet_cityscapes_os16.pth"
    test_image_dir = base_dir / "seg_data" / "testing" / "image"
    test_label_dir = base_dir / "seg_data" / "testing" / "label"
    names_file = base_dir / "seg_data" / "names.txt"
    
    # Load class names
    with open(names_file, 'r') as f:
        class_names = [line.strip() for line in f.readlines()]
    
    num_classes = len(class_names)
    print(f"Number of classes: {num_classes}")
    print(f"Classes: {class_names}")
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model = load_pretrained_model(model_path, num_classes)
    model = model.to(device)
    model.eval()
    
    # Setup transforms
    transform = transforms.Compose([
        transforms.Resize((512, 512)),  # Resize for consistent processing
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    target_transform = transforms.Compose([
        lambda x: torch.from_numpy(np.array(Image.fromarray(x).resize((512, 512), Image.NEAREST))).long()
    ])
    
    # Create dataset and dataloader
    test_dataset = RoadDataset(
        test_image_dir, 
        test_label_dir, 
        transform=transform,
        target_transform=target_transform
    )
    
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    print(f"\nStarting evaluation on {len(test_dataset)} test samples...")
    
    # Evaluation
    all_ious = []
    class_ious_sum = np.zeros(num_classes)
    class_counts = np.zeros(num_classes)
    
    with torch.no_grad():
        for i, (images, targets, filenames) in enumerate(test_loader):
            images = images.to(device)
            targets = targets.to(device)
            
            # Forward pass
            outputs = model(images)
            
            # Get predictions
            preds = torch.argmax(outputs, dim=1)
            
            # Calculate IoU for this batch
            for j in range(images.size(0)):
                pred = preds[j].cpu().numpy()
                target = targets[j].cpu().numpy()
                
                miou, ious = calculate_miou(pred, target, num_classes)
                all_ious.append(miou)
                
                # Accumulate class-wise IoUs
                for class_id, iou in enumerate(ious):
                    if not np.isnan(iou):
                        class_ious_sum[class_id] += iou
                        class_counts[class_id] += 1
                
                if (i + 1) % 10 == 0:
                    print(f"Processed {i+1}/{len(test_loader)} images, Current mIoU: {miou:.4f}")
    
    # Calculate final metrics
    overall_miou = np.mean(all_ious)
    class_mean_ious = np.divide(class_ious_sum, class_counts, 
                                out=np.zeros_like(class_ious_sum), 
                                where=class_counts!=0)
    
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Overall Mean IoU (mIoU): {overall_miou:.4f}")
    print(f"Number of test samples: {len(test_dataset)}")
    
    print("\nClass-wise IoU:")
    print("-" * 40)
    for i, (class_name, iou, count) in enumerate(zip(class_names, class_mean_ious, class_counts)):
        if count > 0:
            print(f"{i:2d}. {class_name:15s}: {iou:.4f} ({int(count)} samples)")
        else:
            print(f"{i:2d}. {class_name:15s}: No samples")
    
    print(f"\nFinal Mean Intersection over Union (mIoU): {overall_miou:.4f}")
    print("="*60)
    
    # Save results
    results_file = base_dir / "test_results_custom_model.txt"
    with open(results_file, 'w') as f:
        f.write("Road Scene Segmentation Test Results (Custom UNetDeepLabFusion Model)\n")
        f.write("="*40 + "\n")
        f.write(f"Model: {model_path.name}\n")
        f.write(f"Architecture: UNet + DeepLabV3+ Fusion with EfficientNet-B0\n")
        f.write(f"Test samples: {len(test_dataset)}\n")
        f.write(f"Overall mIoU: {overall_miou:.4f}\n\n")
        f.write("Class-wise IoU:\n")
        for i, (class_name, iou, count) in enumerate(zip(class_names, class_mean_ious, class_counts)):
            if count > 0:
                f.write(f"{i:2d}. {class_name:15s}: {iou:.4f} ({int(count)} samples)\n")
            else:
                f.write(f"{i:2d}. {class_name:15s}: No samples\n")
    
    print(f"Results saved to: {results_file}")

if __name__ == "__main__":
    main()
