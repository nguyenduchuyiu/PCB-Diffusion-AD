#!/usr/bin/env python3
"""
Simple inference script for PCB Diffusion AD
Usage: python inference.py --model_path path/to/model.pt --image_path path/to/image.jpg
"""

import torch
import cv2
import numpy as np
import argparse
from models.Recon_subnetwork import UNetModel
from models.Seg_subnetwork import SegmentationSubNetwork
from models.DDPM import GaussianDiffusionModel, get_beta_schedule
import matplotlib.pyplot as plt
from collections import defaultdict

def defaultdict_from_json(d):
    """Convert dict to defaultdict"""
    if isinstance(d, defaultdict):
        return d  # Already a defaultdict
    return defaultdict(str, d)

def find_available_models(output_dir="outputs"):
    """Find all available model checkpoints"""
    import os
    import glob
    
    models = []
    pattern = f"{output_dir}/model/diff-params-ARGS=*/*/params-*.pt"
    
    for model_path in glob.glob(pattern):
        try:
            file_size = os.path.getsize(model_path)
            if file_size > 1024:  # At least 1KB
                models.append({
                    'path': model_path,
                    'size_mb': file_size / (1024*1024),
                    'type': 'best' if 'best' in model_path else 'last'
                })
        except:
            continue
    
    return sorted(models, key=lambda x: x['size_mb'], reverse=True)

def load_model(model_path, device='cuda'):
    """Load trained model from checkpoint"""
    print(f"Loading model from {model_path}")
    
    # Check if file exists
    import os
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Check file size
    file_size = os.path.getsize(model_path)
    print(f"Model file size: {file_size / (1024*1024):.2f} MB")
    
    if file_size < 1024:  # Less than 1KB is suspicious
        raise ValueError(f"Model file too small ({file_size} bytes) - likely corrupted")
    
    try:
        # Try loading with weights_only=False for compatibility with defaultdict
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    except Exception as e:
        # If that fails, try with safe_globals for defaultdict
        try:
            print("⚠️  Trying alternative loading method...")
            from collections import defaultdict
            torch.serialization.add_safe_globals([defaultdict])
            checkpoint = torch.load(model_path, map_location=device)
        except Exception as e2:
            print(f"❌ Failed to load model: {e}")
            print("💡 Try these solutions:")
            print("   1. Re-download the model file")
            print("   2. Check if training completed successfully")
            print("   3. Use PyTorch < 2.6 for compatibility")
            raise e
    # Handle both dict and defaultdict args - convert to defaultdict for consistency
    args = defaultdict_from_json(checkpoint['args'])
    
    # Debug: Print loaded args
    print(f"📊 Loaded model config:")
    print(f"   - Epoch: {checkpoint.get('n_epoch', 'unknown')}")
    print(f"   - Image size: {args['img_size']}")
    print(f"   - Channels: {args['channels']}")
    print(f"   - Base channels: {args['base_channels']}")
    print(f"   - Channel mults: {args['channel_mults']}")
    if 'best_loss' in checkpoint:
        print(f"   - Best loss: {checkpoint['best_loss']:.4f} at epoch {checkpoint['best_epoch']}")
    
    # Create models
    in_channels = args["channels"]
    unet_model = UNetModel(
        args['img_size'][0], args['base_channels'], 
        channel_mults=args['channel_mults'], 
        dropout=args["dropout"], 
        n_heads=args["num_heads"], 
        n_head_channels=args["num_head_channels"],
        in_channels=in_channels
    ).to(device)
    
    seg_model = SegmentationSubNetwork(in_channels=6, out_channels=1).to(device)
    
    # Load weights
    unet_model.load_state_dict(checkpoint['unet_model_state_dict'])
    seg_model.load_state_dict(checkpoint['seg_model_state_dict'])
    
    # Setup DDPM
    betas = get_beta_schedule(args['T'], args['beta_schedule'])
    ddpm = GaussianDiffusionModel(
        args['img_size'], betas, 
        loss_weight=args['loss_weight'],
        loss_type=args['loss-type'], 
        noise=args["noise_fn"], 
        img_channels=in_channels
    )
    
    unet_model.eval()
    seg_model.eval()
    
    print(f"✅ Model loaded successfully!")
    return unet_model, seg_model, ddpm, args

def preprocess_image(image_path, img_size=[256, 256]):
    """Load and preprocess image"""
    print(f"Loading image: {image_path}")
    
    # Load image
    image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    original_shape = image.shape[:2]
    
    # Resize
    image = cv2.resize(image, (img_size[1], img_size[0]))
    
    # Normalize to [-1, 1] like in training
    image = (image / 255.0) * 2.0 - 1.0
    
    # Convert to tensor
    image = np.transpose(image.astype(np.float32), (2, 0, 1))
    image = torch.from_numpy(image).unsqueeze(0)  # Add batch dimension
    
    return image, original_shape

def inference(unet_model, seg_model, ddpm, image, args, device='cuda'):
    """Run inference on image"""
    print("Running inference...")
    
    image = image.to(device)
    
    with torch.no_grad():
        # Sample noise timesteps (fixed values like in eval)
        normal_t_tensor = torch.tensor([args["eval_normal_t"]], device=device).repeat(image.shape[0])
        noisier_t_tensor = torch.tensor([args["eval_noisier_t"]], device=device).repeat(image.shape[0])
        
        # Use the same method as in eval.py
        loss, pred_x_0_condition, pred_x_0_normal, pred_x_0_noisier, x_normal_t, x_noisier_t, pred_x_t_noisier = ddpm.norm_guided_one_step_denoising_eval(
            unet_model, image, normal_t_tensor, noisier_t_tensor, args
        )
        
        # Get anomaly mask using the conditioned reconstruction
        anomaly_mask_logits = seg_model(torch.cat((image, pred_x_0_condition), dim=1))
        anomaly_mask = torch.sigmoid(anomaly_mask_logits)
        
        # Calculate anomaly score using the same method as eval.py
        topk_out_mask = torch.flatten(anomaly_mask[0], start_dim=1)
        topk_out_mask = torch.topk(topk_out_mask, 50, dim=1, largest=True)[0]
        anomaly_score = torch.mean(topk_out_mask).item()
        
    return {
        'original': image,
        'recon_normal': pred_x_0_normal,
        'recon_noisier': pred_x_0_noisier,
        'recon_conditioned': pred_x_0_condition,
        'anomaly_mask': anomaly_mask,
        'anomaly_score': anomaly_score
    }

def visualize_results(results, save_path=None):
    """Visualize inference results"""
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    # Convert tensors to numpy for visualization
    original = results['original'].cpu().squeeze().permute(1, 2, 0).numpy()
    recon_conditioned = results['recon_conditioned'].cpu().squeeze().permute(1, 2, 0).numpy()
    recon_noisier = results['recon_noisier'].cpu().squeeze().permute(1, 2, 0).numpy()
    anomaly_mask = results['anomaly_mask'].cpu().squeeze().numpy()
    
    # Normalize from [-1, 1] to [0, 1] for visualization
    original = (original + 1) / 2
    recon_conditioned = (recon_conditioned + 1) / 2
    recon_noisier = (recon_noisier + 1) / 2
    
    # Clip values to [0, 1]
    original = np.clip(original, 0, 1)
    recon_conditioned = np.clip(recon_conditioned, 0, 1)
    recon_noisier = np.clip(recon_noisier, 0, 1)
    
    axes[0].imshow(original)
    axes[0].set_title('Original')
    axes[0].axis('off')
    
    axes[1].imshow(recon_conditioned)
    axes[1].set_title('Reconstruction (Conditioned)')
    axes[1].axis('off')
    
    axes[2].imshow(recon_noisier)
    axes[2].set_title('Reconstruction (Noisier)')
    axes[2].axis('off')
    
    axes[3].imshow(anomaly_mask, cmap='hot')
    axes[3].set_title(f'Anomaly Mask\nScore: {results["anomaly_score"]:.4f}')
    axes[3].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Results saved to: {save_path}")
    else:
        plt.show()

def main():
    
    model_path = "/kaggle/input/testmodel1/pytorch/default/1/params-last.pt"
    image_path = "/kaggle/input/pcb-dataset/RealIAD/PCB5/test/bad/pcb_0001_NG_HS_C1_20231028093757.jpg"
    output_path = "outputs/inference/RealIAD/apple/00000000.jpg"
    
    # Check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Find available models if specified path doesn't work
    import os
    if not os.path.exists(model_path):
        print(f"⚠️  Model not found at: {model_path}")
        print("🔍 Searching for available models...")
        
        available_models = find_available_models()
        if available_models:
            print("\n📋 Available models:")
            for i, model in enumerate(available_models):
                print(f"  {i+1}. {model['path']} ({model['size_mb']:.1f}MB, {model['type']})")
            
            # Use the largest model (likely most complete)
            model_path = available_models[0]['path']
            print(f"\n✅ Using: {model_path}")
        else:
            print("❌ No valid models found!")
            return
    
    try:
        # Load model
        unet_model, seg_model, ddpm, model_args = load_model(model_path, device)
        
        # Load and preprocess image
        image, original_shape = preprocess_image(image_path, model_args['img_size'])
        
        # Run inference
        results = inference(unet_model, seg_model, ddpm, image, model_args, device)
        
        # Show results
        print(f"\n🎯 Anomaly Score: {results['anomaly_score']:.4f}")
        if results['anomaly_score'] > 0.5:
            print("🚨 ANOMALY DETECTED!")
        else:
            print("✅ Normal")
        
        # Create output directory if needed
        if output_path:
            import os
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Visualize
        visualize_results(results, output_path)
        
        # Memory cleanup
        torch.cuda.empty_cache()
        import gc
        gc.collect()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
