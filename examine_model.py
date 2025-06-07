import torch
import numpy as np

def examine_pretrained_model():
    """Examine the structure of the pretrained model"""
    
    model_path = "best_deeplabv3plus_mobilenet_cityscapes_os16.pth"
    
    try:
        # Load with weights_only=False to handle numpy objects
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        print("✓ Model loaded successfully")
        print(f"Checkpoint type: {type(checkpoint)}")
        print(f"Checkpoint keys: {list(checkpoint.keys()) if isinstance(checkpoint, dict) else 'Not a dict'}")
        
        # Extract state dict
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                print("Using 'model_state_dict'")
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
                print("Using 'state_dict'")
            elif 'model_state' in checkpoint:
                state_dict = checkpoint['model_state']
                print("Using 'model_state'")
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
                print("Using 'model'")
            else:
                # Show what's in the checkpoint
                for key, value in checkpoint.items():
                    print(f"  {key}: {type(value)} - {value if not hasattr(value, 'shape') else f'tensor {value.shape}'}")
                return
        else:
            state_dict = checkpoint
            print("Checkpoint is the state_dict directly")
        
        print(f"\nState dict has {len(state_dict)} parameters")
        
        # Show some key parameter names and shapes
        print("\nKey parameters:")
        count = 0
        for name, param in state_dict.items():
            if count < 20:  # Show first 20 parameters
                print(f"  {name}: {param.shape}")
                count += 1
            
            # Look for classifier/decoder layers
            if any(keyword in name.lower() for keyword in ['classifier', 'decode', 'head', 'final', 'out']):
                print(f"  *** {name}: {param.shape}")
        
        # Check if this looks like a standard DeepLabV3+ structure
        mobilenet_keys = [k for k in state_dict.keys() if 'backbone' in k or 'encoder' in k]
        decoder_keys = [k for k in state_dict.keys() if 'decoder' in k or 'classifier' in k]
        
        print(f"\nBackbone/Encoder parameters: {len(mobilenet_keys)}")
        print(f"Decoder/Classifier parameters: {len(decoder_keys)}")
        
        if mobilenet_keys:
            print("Sample backbone keys:", mobilenet_keys[:5])
        if decoder_keys:
            print("Sample decoder keys:", decoder_keys[:5])
            
    except Exception as e:
        print(f"Error examining model: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    examine_pretrained_model()
