import io
import os
import json
import torch
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from PIL import Image
from utils.lem_utils import CLIPRelevance
import configargparse

def draw_rele_distrib(rele, kde=True):
    rele = rele.view(-1).detach().to("cpu").numpy()
    
    plt.figure()
    if kde:
        sns.kdeplot(rele, color='blue', label='rele')
    else:
        plt.hist(rele, bins=30, color='blue', alpha=0.5, label='rele')
    plt.legend(loc='upper right')
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img = Image.open(buf)
    
    plt.close()
    return img

def load_text_config(config_path):
    """
    Load text configuration from JSON file
    Returns dictionary mapping main text to its helping positives and negatives
    """
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    text_config = {}
    for i, main_text in enumerate(config['text']):
        text_config[main_text] = {
            'helping_positives': config['helping_positives'][i] if 'helping_positives' in config else None,
            'negatives': config['negatives'][i] if 'negatives' in config else None
        }
    return text_config

def process_clip_features(features_dir, output_dir, text_config, alpha=0.5, scale=100, device="cuda"):
    """
    Process saved CLIP features to generate relevancy maps and segmentations
    """
    clip_rele = CLIPRelevance(device=device)
    
    # Create output directories
    pred_segs_pth = os.path.join(output_dir, "pred_segs")
    rele_pth = os.path.join(output_dir, "relevancy")
    
    os.makedirs(pred_segs_pth, exist_ok=True)
    os.makedirs(rele_pth, exist_ok=True)
    
    # Process each saved feature file
    feature_files = [f for f in os.listdir(features_dir) if f.endswith('.pt')]
    for feature_file in feature_files:
        image_name = os.path.splitext(feature_file)[0]
        # Load CLIP features
        # clip_features = torch.load(os.path.join(features_dir, feature_file)).to(device)
        try:
            clip_features = torch.load(os.path.join(features_dir, feature_file), map_location=device)
        except Exception as e:
            print(f"Error loading file {feature_file}: {e}")
            continue
        
        # Create output subdirectories for this image
        os.makedirs(f"{pred_segs_pth}/{image_name}", exist_ok=True)
        # os.makedirs(f"{pred_segs_pth}/{image_name}/distr", exist_ok=True)
        os.makedirs(f"{rele_pth}/{image_name}/array", exist_ok=True)
        os.makedirs(f"{rele_pth}/{image_name}/images", exist_ok=True)
        
        seg_indices = -1 * torch.ones(clip_features.shape[:-1], device=device)
        
        # Process each text prompt and its associated terms
        for i, (main_text, config) in enumerate(text_config.items()):
            # Get relevancy scores using the exact method signature
            rele = clip_rele.get_relevancy(
                clip_features,
                main_text,
                helping_positives=None,#config['helping_positives'],
                negatives=config['negatives'],
                scale=scale
            ).squeeze()[..., 0]
            
            # Generate and save visualizations
            # rele_distr_img = draw_rele_distrib(rele)
            mask = (rele >= alpha)
            
            # Save outputs
            np.save(f"{rele_pth}/{image_name}/array/{main_text}.npy", rele.detach().cpu().numpy())
            torchvision.utils.save_image(rele, f"{rele_pth}/{image_name}/images/{main_text}.png")
            torchvision.utils.save_image(mask.float(), f"{pred_segs_pth}/{image_name}/{main_text}.png")
            # rele_distr_img.save(f"{pred_segs_pth}/{image_name}/distr/{main_text}.png")
            
            seg_indices[mask] = i
        
        # Save text configuration for reference
        with open(f"{pred_segs_pth}/text_config.json", "w") as f:
            json.dump(text_config, f, indent=4)
        
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = configargparse.ArgParser(description="Process saved CLIP features")
    
    # Add config file parameter
    parser.add('--config', required=False, is_config_file=True, help='config file path')
    
    # Add only the necessary parameters
    parser.add_argument('--source_path', type=str, required=True, help='Source path containing clip_features directory')
    parser.add_argument('--input_command', type=str, required=True, help='Path to input command JSON file')
    parser.add_argument('--alpha', type=float, default=0.5, help='Threshold for segmentation')
    parser.add_argument('--scale', type=float, default=10, help='Scale factor for relevancy')
    parser.add_argument('--com_type', type=str, default='argmax', help='Computation type for output directory naming')
    
    args = parser.parse_args()
    
    # Load text configuration from the input command JSON
    text_config = load_text_config(args.input_command)
    
    # Construct features directory from source path
    features_dir = args.source_path
    
    output_dir = os.path.join('Neg',os.path.splitext(args.input_command)[0])

    
    process_clip_features(
        features_dir,
        output_dir,
        text_config,
        args.alpha,
        args.scale
    )
    
    print("Processing completed.")

    # example usage:
    # python3 renderfromclip.py --config Qwen/rele_configs/scene_036_30_forward.cfg