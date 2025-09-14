from random import seed
import torch
import os
import json
import matplotlib.pyplot as plt
import numpy as np
from torch import optim
from torch.utils.data import DataLoader
from models.Recon_subnetwork import UNetModel, update_ema_params
from models.Seg_subnetwork import SegmentationSubNetwork
from tqdm import tqdm
import torch.nn as nn
from data.dataset_beta_thresh import RealIADTrainDataset, RealIADTestDataset
from math import exp
import torch.nn.functional as F
from torch.amp import autocast
from torch.cuda.amp import GradScaler
from models.DDPM import GaussianDiffusionModel, get_beta_schedule
from scipy.ndimage import gaussian_filter
from skimage.measure import label, regionprops
from sklearn.metrics import roc_auc_score,auc,average_precision_score
import pandas as pd
from collections import defaultdict
import time
import psutil
import gc
from utils.profiler import PerformanceProfiler

def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)    

def defaultdict_from_json(jsonDict):
    func = lambda: defaultdict(str)
    dd = func()
    dd.update(jsonDict)
    return dd

class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=4, logits=False, reduce=True): 
        super(BinaryFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        # Always use binary_cross_entropy_with_logits for mixed precision compatibility
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss

def count_parameters(model):
    """Count the number of trainable parameters in a model"""
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params

def print_model_info(model, model_name):
    """Print detailed information about model parameters"""
    total_params = count_parameters(model)
    print(f"{model_name}: {total_params:,} trainable parameters")
    
    # Memory estimation (FP32)
    param_memory_mb = total_params * 4 / (1024 ** 2)
    print(f"{model_name} Memory: ~{param_memory_mb:.1f} MB")
    
    return total_params

def plot_learning_curves(train_loss_list, train_noise_loss_list, train_focal_loss_list, 
                        train_smL1_loss_list, loss_x_list, image_auroc_list, 
                        pixel_auroc_list, performance_x_list, sub_class, args, inline=False):
    """Plot simple loss curves only - minimal memory usage"""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Plot losses only
    if len(loss_x_list) > 0:
        ax.plot(loss_x_list, train_loss_list, 'b-', label='Total Loss', linewidth=2)
        ax.plot(loss_x_list, train_noise_loss_list, 'r-', label='Noise Loss')
        ax.plot(loss_x_list, train_focal_loss_list, 'g-', label='Focal Loss')
        ax.plot(loss_x_list, train_smL1_loss_list, 'm-', label='SmoothL1 Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        current_epoch = loss_x_list[-1] if loss_x_list else 0
        current_loss = train_loss_list[-1] if train_loss_list else 0
        ax.set_title(f'{sub_class} - Training Losses (Epoch {current_epoch}, Loss: {current_loss:.4f})')
        ax.legend()
        ax.grid(True)
        ax.set_yscale('log')
    else:
        ax.text(0.5, 0.5, 'No loss data yet', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Training Losses (Waiting for data...)')
    
    plt.tight_layout()
    
    if inline:
        # Display inline for Jupyter notebooks
        from IPython.display import display, clear_output
        clear_output(wait=True)
        display(fig)
        plt.close()
        # Print update
        if len(loss_x_list) > 0:
            current_epoch = loss_x_list[-1] if loss_x_list else 0
            current_loss = train_loss_list[-1] if train_loss_list else 0
            print(f"📊 Loss tracking: Epoch {current_epoch}, Loss: {current_loss:.4f}")
    else:
        # Save the plot to file
        os.makedirs(f'{args["output_path"]}/learning_curves/ARGS={args["arg_num"]}', exist_ok=True)
        plt.savefig(f'{args["output_path"]}/learning_curves/ARGS={args["arg_num"]}/{sub_class}_learning_curves.png', 
                    dpi=300, bbox_inches='tight')
        plt.close()
        
        # Print update
        if len(loss_x_list) > 0:
            current_epoch = loss_x_list[-1] if loss_x_list else 0
            current_loss = train_loss_list[-1] if train_loss_list else 0
            print(f"📊 Loss tracking: Epoch {current_epoch}, Loss: {current_loss:.4f}")
            print(f"   📁 Saved to: outputs/learning_curves/ARGS={args['arg_num']}/{sub_class}_learning_curves.png")

def is_jupyter_environment():
    """Check if running in Jupyter notebook environment"""
    try:
        from IPython import get_ipython
        return get_ipython() is not None and get_ipython().__class__.__name__ == 'ZMQInteractiveShell'
    except ImportError:
        return False

def monitor_system_resources():
    """Monitor system resources"""
    # CPU usage
    cpu_percent = psutil.cpu_percent(interval=1)
    
    # Memory usage
    memory = psutil.virtual_memory()
    memory_percent = memory.percent
    memory_available_gb = memory.available / (1024**3)
    
    # GPU memory (if available)
    gpu_memory_info = ""
    if torch.cuda.is_available():
        gpu_memory_allocated = torch.cuda.memory_allocated() / (1024**3)
        gpu_memory_reserved = torch.cuda.memory_reserved() / (1024**3)
        gpu_memory_info = f" | GPU: {gpu_memory_allocated:.2f}GB allocated, {gpu_memory_reserved:.2f}GB reserved"
    
    return f"CPU: {cpu_percent:.1f}% | RAM: {memory_percent:.1f}% ({memory_available_gb:.2f}GB free){gpu_memory_info}"

def train(training_dataset_loader, testing_dataset_loader, args, data_len,sub_class,class_type,device, num_gpus=1):
    
    # Check if running in Jupyter environment
    use_inline_plots = is_jupyter_environment()
    if use_inline_plots:
        print("📊 Detected Jupyter environment - using inline plots to save memory")
    else:
        print("📊 Using file-based plots")
    
    # Check for resume training
    resume_from_checkpoint = args.get('resume_training', False)
    start_epoch = 0
    
    # Initialize profiler
    profiler = PerformanceProfiler(log_dir=f'{args["output_path"]}/profiling/ARGS={args["arg_num"]}/{sub_class}')
    
    in_channels = args["channels"]
    unet_model = UNetModel(args['img_size'][0], args['base_channels'], channel_mults=args['channel_mults'], dropout=args[
                "dropout"], n_heads=args["num_heads"], n_head_channels=args["num_head_channels"],
            in_channels=in_channels
            ).to(device)


    betas = get_beta_schedule(args['T'], args['beta_schedule'])

    ddpm_sample =  GaussianDiffusionModel(
            args['img_size'], betas, loss_weight=args['loss_weight'],
            loss_type=args['loss-type'], noise=args["noise_fn"], img_channels=in_channels
            )

    seg_model=SegmentationSubNetwork(in_channels=6, out_channels=1).to(device)

    # Enable multi-GPU training if available
    if num_gpus > 1:
        print(f"Wrapping models with DataParallel for {num_gpus} GPUs")
        unet_model = torch.nn.DataParallel(unet_model)
        seg_model = torch.nn.DataParallel(seg_model)

    # Count trainable parameters
    unet_params = count_parameters(unet_model)
    seg_params = count_parameters(seg_model)
    total_params = unet_params + seg_params
    
    print(f"Model Parameters:")
    print(f"  - UNet Model: {unet_params:,} parameters")
    print(f"  - Segmentation Model: {seg_params:,} parameters") 
    print(f"  - Total Trainable Parameters: {total_params:,} parameters")
    print(f"  - Memory (approx): {total_params * 4 / 1024**2:.1f} MB (FP32)")

    optimizer_ddpm = optim.Adam( unet_model.parameters(), lr=args['diffusion_lr'],weight_decay=args['weight_decay'])
    
    optimizer_seg = optim.Adam(seg_model.parameters(),lr=args['seg_lr'],weight_decay=args['weight_decay'])
    
    # Initialize mixed precision scaler
    use_mixed_precision = args.get('use_mixed_precision', True) and torch.cuda.is_available()
    scaler = GradScaler() if use_mixed_precision else None
    
    # Gradient accumulation settings
    gradient_accumulation_steps = args.get('gradient_accumulation_steps', 1)
    effective_batch_size = args['Batch_Size'] * gradient_accumulation_steps
    
    print(f"Mixed Precision (FP16): {'Enabled' if use_mixed_precision else 'Disabled'}")
    print(f"Gradient Accumulation Steps: {gradient_accumulation_steps}")
    print(f"Effective Batch Size: {effective_batch_size}")

    loss_focal = BinaryFocalLoss().to(device)
    loss_smL1= nn.SmoothL1Loss().to(device)
    

    scheduler_seg =optim.lr_scheduler.CosineAnnealingLR(optimizer_seg, T_max=10, eta_min=0, last_epoch=- 1, verbose=False)
    
    # Initialize training variables
    train_loss_list=[]
    train_noise_loss_list=[]
    train_focal_loss_list=[]
    train_smL1_loss_list=[]
    loss_x_list=[]
    best_image_auroc=0.0
    best_pixel_auroc=0.0
    best_epoch=0
    image_auroc_list=[]
    pixel_auroc_list=[]
    performance_x_list=[]
    
    # Resume training if checkpoint exists
    if resume_from_checkpoint:
        checkpoint_path = f'{args["output_path"]}/model/diff-params-ARGS={args["arg_num"]}/{sub_class}/params-last.pt'
        if os.path.exists(checkpoint_path):
            print(f"🔄 Resuming training from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            
            # Load model states
            if hasattr(unet_model, 'module'):
                unet_model.module.load_state_dict(checkpoint['unet_model_state_dict'])
                seg_model.module.load_state_dict(checkpoint['seg_model_state_dict'])
            else:
                unet_model.load_state_dict(checkpoint['unet_model_state_dict'])
                seg_model.load_state_dict(checkpoint['seg_model_state_dict'])
            
            start_epoch = checkpoint['n_epoch']
            print(f"🎯 Resuming from epoch {start_epoch}")
            
            # Load training history if available
            if 'train_loss_list' in checkpoint:
                train_loss_list = checkpoint.get('train_loss_list', [])
                train_noise_loss_list = checkpoint.get('train_noise_loss_list', [])
                train_focal_loss_list = checkpoint.get('train_focal_loss_list', [])
                train_smL1_loss_list = checkpoint.get('train_smL1_loss_list', [])
                loss_x_list = checkpoint.get('loss_x_list', [])
                image_auroc_list = checkpoint.get('image_auroc_list', [])
                pixel_auroc_list = checkpoint.get('pixel_auroc_list', [])
                performance_x_list = checkpoint.get('performance_x_list', [])
                best_image_auroc = checkpoint.get('best_image_auroc', 0.0)
                best_pixel_auroc = checkpoint.get('best_pixel_auroc', 0.0)
                best_epoch = checkpoint.get('best_epoch', 0)
                print(f"📊 Loaded training history: {len(train_loss_list)} epochs")
        else:
            print(f"⚠️  No checkpoint found at {checkpoint_path}, starting from scratch")
            start_epoch = 0
    
    tqdm_epoch = range(start_epoch, args['EPOCHS'])
    
    # Performance monitoring
    epoch_times = []
    resource_monitor_interval = 100  # Monitor resources every 100 epochs
    
    for epoch in tqdm_epoch:
        epoch_start_time = time.time()
        unet_model.train()
        seg_model.train()
        train_loss = 0.0
        train_focal_loss=0.0
        train_smL1_loss = 0.0
        train_noise_loss = 0.0
        tbar = tqdm(training_dataset_loader)
        
        # Initialize gradient accumulation
        optimizer_ddpm.zero_grad()
        optimizer_seg.zero_grad()
        
        for i, sample in enumerate(tbar):
            batch_start_time = time.time()
            
            # Profile data loading
            with profiler.timer('data_loading'):
                aug_image=sample['augmented_image'].to(device)
                anomaly_mask = sample["anomaly_mask"].to(device)
                anomaly_label = sample["has_anomaly"].to(device).squeeze()

            # Profile forward pass
            with profiler.timer('forward_pass'):
                # Mixed precision forward pass
                if use_mixed_precision:
                    with autocast('cuda'):
                        noise_loss, pred_x0,normal_t,x_normal_t,x_noiser_t = ddpm_sample.norm_guided_one_step_denoising(unet_model, aug_image, anomaly_label,args)
                        pred_mask = seg_model(torch.cat((aug_image, pred_x0), dim=1)) 

                        #loss
                        focal_loss = loss_focal(pred_mask,anomaly_mask)
                        smL1_loss = loss_smL1(pred_mask, anomaly_mask)
                        loss = (noise_loss + 5*focal_loss + smL1_loss) / gradient_accumulation_steps
                else:
                    noise_loss, pred_x0,normal_t,x_normal_t,x_noiser_t = ddpm_sample.norm_guided_one_step_denoising(unet_model, aug_image, anomaly_label,args)
                    pred_mask = seg_model(torch.cat((aug_image, pred_x0), dim=1)) 

                    #loss
                    focal_loss = loss_focal(pred_mask,anomaly_mask)
                    smL1_loss = loss_smL1(pred_mask, anomaly_mask)
                    loss = (noise_loss + 5*focal_loss + smL1_loss) / gradient_accumulation_steps
            
            # Profile backward pass
            with profiler.timer('backward_pass'):
                # Mixed precision backward pass
                if use_mixed_precision:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

            # Profile optimizer step
            with profiler.timer('optimizer_step'):
                # Gradient accumulation step
                if (i + 1) % gradient_accumulation_steps == 0:
                    if use_mixed_precision:
                        scaler.step(optimizer_ddpm)
                        scaler.step(optimizer_seg)
                        scaler.update()
                    else:
                        optimizer_ddpm.step()
                        optimizer_seg.step()
                    
                    scheduler_seg.step()
                    optimizer_ddpm.zero_grad()
                    optimizer_seg.zero_grad()

            train_loss += loss.item() * gradient_accumulation_steps  # Scale back for logging
            tbar.set_description('Epoch:%d, Train loss: %.3f' % (epoch, train_loss))

            train_smL1_loss += smL1_loss.item()
            train_focal_loss+=5*focal_loss.item()
            train_noise_loss+=noise_loss.item()
            
            # Log batch metrics
            batch_time = time.time() - batch_start_time
            profiler.log_training_metrics(
                epoch=epoch,
                batch_idx=i,
                losses={
                    'total_loss': loss.item() * gradient_accumulation_steps,
                    'noise_loss': noise_loss.item(),
                    'focal_loss': focal_loss.item(),
                    'smL1_loss': smL1_loss.item()
                },
                batch_time=batch_time
            )
            
        # Record epoch time
        epoch_time = time.time() - epoch_start_time
        epoch_times.append(epoch_time)
        
        # Log system metrics and monitor resources
        profiler.log_system_metrics()
        profiler.log_training_metrics(epoch=epoch, batch_idx=0, losses={}, epoch_time=epoch_time)
        
        # Monitor system resources periodically
        if epoch % resource_monitor_interval == 0 and epoch > 0:
            resource_info = monitor_system_resources()
            avg_epoch_time = np.mean(epoch_times[-resource_monitor_interval:]) if len(epoch_times) >= resource_monitor_interval else np.mean(epoch_times)
            print(f"Epoch {epoch} | {resource_info} | Avg Epoch Time: {avg_epoch_time:.2f}s")
            
            # Check for bottlenecks
            bottlenecks = profiler.detect_bottlenecks()
            if bottlenecks:
                print("⚠️  Performance bottlenecks detected:")
                for bottleneck in bottlenecks:
                    print(f"   - {bottleneck}")
            
            # Clear GPU cache periodically
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()

        # Save losses every epoch (always track progress)
        train_loss_list.append(round(train_loss,3))
        train_smL1_loss_list.append(round(train_smL1_loss,3))
        train_focal_loss_list.append(round(train_focal_loss,3))
        train_noise_loss_list.append(round(train_noise_loss,3))
        loss_x_list.append(int(epoch))
        
        # Plot learning curves every epoch for first 10 epochs, then every 10 epochs
        if epoch < 10 or epoch % 10 == 0:
            plot_learning_curves(train_loss_list, train_noise_loss_list, train_focal_loss_list, 
                                train_smL1_loss_list, loss_x_list, image_auroc_list, 
                                pixel_auroc_list, performance_x_list, sub_class, args, inline=use_inline_plots)


        if (epoch+1) % 10==0 and epoch > 0:
            if data_len > 0:  # Only evaluate if test data exists
                temp_image_auroc,temp_pixel_auroc= eval(testing_dataset_loader,args,unet_model,seg_model,data_len,sub_class,device)
                image_auroc_list.append(temp_image_auroc)
                pixel_auroc_list.append(temp_pixel_auroc)
                performance_x_list.append(int(epoch))
                if(temp_image_auroc+temp_pixel_auroc>=best_image_auroc+best_pixel_auroc):
                    if temp_image_auroc>=best_image_auroc:
                        training_history = {
                            'train_loss_list': train_loss_list,
                            'train_noise_loss_list': train_noise_loss_list,
                            'train_focal_loss_list': train_focal_loss_list,
                            'train_smL1_loss_list': train_smL1_loss_list,
                            'loss_x_list': loss_x_list,
                            'image_auroc_list': image_auroc_list,
                            'pixel_auroc_list': pixel_auroc_list,
                            'performance_x_list': performance_x_list,
                            'best_image_auroc': temp_image_auroc,
                            'best_pixel_auroc': temp_pixel_auroc,
                            'best_epoch': epoch
                        }
                        save(unet_model,seg_model, args=args,final='best',epoch=epoch,sub_class=sub_class, training_history=training_history)
                        best_image_auroc = temp_image_auroc
                        best_pixel_auroc = temp_pixel_auroc
                        best_epoch = epoch
            else:
                print(f"⚠️  Skipping evaluation at epoch {epoch+1} - no test data available")
                
            
    # Save final checkpoint with training history
    final_training_history = {
        'train_loss_list': train_loss_list,
        'train_noise_loss_list': train_noise_loss_list,
        'train_focal_loss_list': train_focal_loss_list,
        'train_smL1_loss_list': train_smL1_loss_list,
        'loss_x_list': loss_x_list,
        'image_auroc_list': image_auroc_list,
        'pixel_auroc_list': pixel_auroc_list,
        'performance_x_list': performance_x_list,
        'best_image_auroc': best_image_auroc,
        'best_pixel_auroc': best_pixel_auroc,
        'best_epoch': best_epoch
    }
    save(unet_model,seg_model, args=args,final='last',epoch=args['EPOCHS']-1,sub_class=sub_class, training_history=final_training_history)

    # Skip final plot to save memory
    
    # Save training statistics
    training_stats = {
        'total_epochs': args['EPOCHS'],
        'best_epoch': best_epoch,
        'best_image_auroc': best_image_auroc,
        'best_pixel_auroc': best_pixel_auroc,
        'avg_epoch_time': np.mean(epoch_times) if epoch_times else 0,
        'total_training_time': sum(epoch_times) if epoch_times else 0
    }
    
    with open(f'{args["output_path"]}/metrics/ARGS={args["arg_num"]}/{sub_class}_training_stats.json', 'w') as f:
        json.dump(training_stats, f, indent=2)

    # Save profiler results (text only, no plots)
    profiler.print_summary()
    profiler.save_stats()

    temp = {"classname":[sub_class],"Image-AUROC": [best_image_auroc],"Pixel-AUROC":[best_pixel_auroc],"epoch":best_epoch}
    df_class = pd.DataFrame(temp)
    df_class.to_csv(f"{args['output_path']}/metrics/ARGS={args['arg_num']}/{args['eval_normal_t']}_{args['eval_noisier_t']}t_{args['condition_w']}_{class_type}_image_pixel_auroc_train.csv", mode='a',header=False,index=False)
   
    

def eval(testing_dataset_loader,args,unet_model,seg_model,data_len,sub_class,device):
    unet_model.eval()
    seg_model.eval()
    os.makedirs(f'{args["output_path"]}/metrics/ARGS={args["arg_num"]}/{sub_class}/', exist_ok=True)
    in_channels = args["channels"]
    betas = get_beta_schedule(args['T'], args['beta_schedule'])

    ddpm_sample =  GaussianDiffusionModel(
            args['img_size'], betas, loss_weight=args['loss_weight'],
            loss_type=args['loss-type'], noise=args["noise_fn"], img_channels=in_channels
            )
    
    print("data_len",data_len)
    total_image_pred = np.array([])
    total_image_gt =np.array([])
    total_pixel_gt=np.array([])
    total_pixel_pred = np.array([])
    tbar = tqdm(testing_dataset_loader)
    for i, sample in enumerate(tbar):
        image = sample["image"].to(device)
        target=sample['has_anomaly'].to(device)
        gt_mask = sample["mask"].to(device)

        normal_t_tensor = torch.tensor([args["eval_normal_t"]], device=image.device).repeat(image.shape[0])
        noiser_t_tensor = torch.tensor([args["eval_noisier_t"]], device=image.device).repeat(image.shape[0])
        loss,pred_x_0_condition,pred_x_0_normal,pred_x_0_noisier,x_normal_t,x_noiser_t,pred_x_t_noisier = ddpm_sample.norm_guided_one_step_denoising_eval(unet_model, image, normal_t_tensor,noiser_t_tensor,args)
        pred_mask = seg_model(torch.cat((image, pred_x_0_condition), dim=1)) 

        out_mask = pred_mask

        topk_out_mask = torch.flatten(out_mask[0], start_dim=1)
        topk_out_mask = torch.topk(topk_out_mask, 50, dim=1, largest=True)[0]
        image_score = torch.mean(topk_out_mask)
        
        total_image_pred=np.append(total_image_pred,image_score.detach().cpu().numpy())
        total_image_gt=np.append(total_image_gt,target[0].detach().cpu().numpy())


        flatten_pred_mask=out_mask[0].flatten().detach().cpu().numpy()
        flatten_gt_mask =gt_mask[0].flatten().detach().cpu().numpy().astype(int)
            
        total_pixel_gt=np.append(total_pixel_gt,flatten_gt_mask)
        total_pixel_pred=np.append(total_pixel_pred,flatten_pred_mask)
        
        
    print(sub_class)
    auroc_image = round(roc_auc_score(total_image_gt,total_image_pred),3)*100
    print("Image AUC-ROC: " ,auroc_image)
    
    auroc_pixel =  round(roc_auc_score(total_pixel_gt, total_pixel_pred),3)*100
    print("Pixel AUC-ROC:" ,auroc_pixel)
   
    return auroc_image,auroc_pixel


def save(unet_model,seg_model, args,final,epoch,sub_class, training_history=None):
    
    # Handle DataParallel models - save the underlying module
    unet_state_dict = unet_model.module.state_dict() if hasattr(unet_model, 'module') else unet_model.state_dict()
    seg_state_dict = seg_model.module.state_dict() if hasattr(seg_model, 'module') else seg_model.state_dict()
    
    # Base checkpoint data
    checkpoint_data = {
        'n_epoch':              epoch,
        'unet_model_state_dict': unet_state_dict,
        'seg_model_state_dict':  seg_state_dict,
        "args":                 args
    }
    
    # Add training history for resume functionality
    if training_history is not None:
        checkpoint_data.update(training_history)
    
    torch.save(checkpoint_data, f'{args["output_path"]}/model/diff-params-ARGS={args["arg_num"]}/{sub_class}/params-{final}.pt')
    
    

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Check for multiple GPUs
    num_gpus = torch.cuda.device_count()
    print(f"Number of available GPUs: {num_gpus}")
    if num_gpus > 1:
        print(f"Using {num_gpus} GPUs for training")
    elif num_gpus == 1:
        print("Using single GPU for training")
    else:
        print("Using CPU for training")
    
    # read file from argument
    file = "args1.json"
    # load the json args
    with open(f'./args/{file}', 'r') as f:
        args = json.load(f)
    args['arg_num'] = file[4:-5]
    args = defaultdict_from_json(args)

    real_iad_classes = os.listdir(os.path.join(args["data_root_path"], args['data_name']))

    for sub_class in real_iad_classes:   
        print("class", sub_class)
        
        subclass_path = os.path.join(args["data_root_path"], args['data_name'], sub_class)
        
        training_dataset = RealIADTrainDataset(
            subclass_path, sub_class, img_size=args["img_size"], args=args
        )
        testing_dataset = RealIADTestDataset(
            subclass_path, sub_class, img_size=args["img_size"]
        )
        class_type=args['data_name']
        

        print(file, args)     

        data_len = len(testing_dataset)
        
        # Check if test dataset is empty
        if data_len == 0:
            print(f"⚠️  WARNING: Test dataset for {sub_class} is empty!")
            print(f"   - Training will continue but no evaluation will be performed")
            print(f"   - Check data path: {args['data_root_path']}/{args['data_name']}/{sub_class}")
        
        # Calculate effective batch size considering multi-GPU and gradient accumulation
        base_batch_size = args['Batch_Size']
        gradient_accumulation_steps = args.get('gradient_accumulation_steps', 1)
        
        # For DataLoader, we use the base batch size
        dataloader_batch_size = base_batch_size
            
        # Total effective batch size
        total_effective_batch_size = dataloader_batch_size * gradient_accumulation_steps
        
        print(f"Batch size configuration:")
        print(f"  - Base batch size: {base_batch_size}")
        print(f"  - DataLoader batch size: {dataloader_batch_size} ({'Multi-GPU' if num_gpus > 1 else 'Single-GPU'})")
        print(f"  - Gradient accumulation steps: {gradient_accumulation_steps}")
        print(f"  - Total effective batch size: {total_effective_batch_size}")

        
        # Optimize DataLoader settings
        optimal_num_workers = min(8, os.cpu_count() // 2) if os.cpu_count() else 4
        print(f"Using {optimal_num_workers} workers for data loading")
        
        training_dataset_loader = DataLoader(
            training_dataset, 
            batch_size=dataloader_batch_size,
            shuffle=True,
            num_workers=optimal_num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=True,  # Keep workers alive between epochs
            prefetch_factor=2  # Prefetch 2 batches per worker
        )
        test_loader = DataLoader(
            testing_dataset, 
            batch_size=1,
            shuffle=False, 
            num_workers=min(4, optimal_num_workers),
            pin_memory=True,
            persistent_workers=True
        )

        # make arg specific directories
        for i in [f'{args["output_path"]}/model/diff-params-ARGS={args["arg_num"]}/{sub_class}',
                f'{args["output_path"]}/diffusion-training-images/ARGS={args["arg_num"]}/{sub_class}',
                 f'{args["output_path"]}/metrics/ARGS={args["arg_num"]}/{sub_class}']:
            try:
                os.makedirs(i)
            except OSError:
                pass

        train(training_dataset_loader, test_loader, args, data_len,sub_class,class_type,device, num_gpus)

if __name__ == '__main__':
    
    seed(42)
    main()
