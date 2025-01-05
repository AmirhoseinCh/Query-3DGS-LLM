import os
import json
import torch
import numpy as np
import torch.nn.functional as F
import torchvision.transforms.functional as tf
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import configargparse

def is_pic(fname):
    return fname.split(".")[-1] in ["JPG", "jpg", "png", "npy"]

def mean_iou(mask1, mask2):
    intersection = torch.logical_and(mask1, mask2).float().sum()
    union = torch.logical_or(mask1, mask2).float().sum()
    iou = intersection / (union + 1e-6)  # Adding a small value to avoid division by zero
    return iou

def accuracy(mask1, mask2):
    correct_predictions = torch.eq(mask1, mask2).float().sum()
    total_pixels = mask1.numel()
    accuracy = correct_predictions / total_pixels
    return accuracy

def precision(mask1, mask2):
    tp = torch.logical_and(mask1, mask2).float().sum()  # True positives
    fp = torch.logical_and(mask1, 1-mask2).float().sum()  # False positives
    precision_value = tp / (tp + fp + 1e-6)  # Adding a small value to avoid division by zero
    return precision_value

def recall(mask1, mask2):
    tp = torch.logical_and(mask1, mask2).float().sum()
    fn = torch.logical_and(1 - mask1, mask2).float().sum()
    recall_value = tp / (tp + fn + 1e-6)
    return recall_value

def read_class_info(json_path):
    with open(json_path, 'r') as f:
        class_info = json.load(f)
    return class_info['text'], class_info['segformer_class_id']

def read_3dovs_masks(renders_dir, gt_dir, class_info):
    renders = {}
    gts = {}
    names = {}
    
    texts, class_ids = class_info
    
    for pic_dir in os.listdir(renders_dir):
        if not os.path.isdir(os.path.join(renders_dir, pic_dir)):
            continue
        
        render_masks = {}
        gt_masks = {}
        image_names = []

        
        render_path = renders_dir / pic_dir
        gt_path = gt_dir / pic_dir
        
        gt = np.load(str(gt_path) + ".npy")

        
        for fname in os.listdir(render_path):
            if not is_pic(fname):
                continue
            
            render = Image.open(render_path / fname).convert('L')
            render = np.array(render) / 255.0
            
            text = fname.split('.')[0]
            if text not in texts:
                continue
            
            class_id = class_ids[texts.index(text)]
            if class_id == -1:
                continue
            
            render_mask = torch.from_numpy(render).float().unsqueeze(0)
            gt_mask = torch.from_numpy((gt == class_id).astype(np.float32)).unsqueeze(0)
            
            render_masks[text] = render_mask
            gt_masks[text] = gt_mask
            
            image_names.append(text)

        
        renders[pic_dir] = render_masks
        gts[pic_dir] = gt_masks
        names[pic_dir] = image_names
    
    return renders, gts, names

def mAP_evaluate(texts, class_ids, relevancy_dir, gt_dir, json_pth=None):
    threshold_values = np.arange(0.0, 1.01, 0.01)   
    picture_AP_list = [] 
    picture_AP_dic = {}
    for pic_dir in tqdm(os.listdir(relevancy_dir), desc="mAP evaluation progress"):

        class_AP_list = []
        class_AP_dic = {}
        if not os.path.isdir(os.path.join(relevancy_dir, pic_dir)):
            print("not a dir")
            continue
        for text, class_id in zip(texts, class_ids):
            if class_id == -1:
                continue
            recall_list = []
            precision_list = []
            
            render = np.load(relevancy_dir / pic_dir / Path("array") / f"{text}.npy")
            gt = np.load(gt_dir / f"{pic_dir}.npy")
            h, w = render.shape[0], render.shape[1]
            


            render = torch.from_numpy(render).float()
            gt = torch.from_numpy((gt == class_id).astype(np.float32))
            
            for threshold in threshold_values:
                msk = (render > threshold).long()
                precision_value = precision(msk, gt)
                recall_value = recall(msk, gt)
                recall_list.append(recall_value)
                precision_list.append(precision_value)
            
            interpolated_recall_levels = np.arange(0.0, 1.01, 0.01)
            AP = 0
            precision_list = np.array(precision_list)
            recall_list = np.array(recall_list)
            for r in interpolated_recall_levels:
                precisions_at_recall_level = precision_list[recall_list >= r]
                if len(precisions_at_recall_level) > 0:
                    interpolated_precision = np.max(precisions_at_recall_level)
                else:
                    interpolated_precision = 0
                AP += interpolated_precision
            AP /= 100
            class_AP_list.append(AP)
            class_AP_dic[text] = AP
        picture_AP = np.mean(class_AP_list)
        picture_AP_list.append(picture_AP)
        picture_AP_dic[pic_dir] = {"pic_mAP":picture_AP, "class_AP":class_AP_dic}
    mAP = np.mean(picture_AP_list)
    if json_pth:
        with open(json_pth, "w") as f:
            json.dump({"mAP": mAP, "detail": picture_AP_dic}, f, indent=4)
    print("  mAP : {:>12.7f}".format(mAP, ".5"))

    return mAP

def lem_evaluate(renders, gts, json_pth=None):
    IoUs = {}
    accuracies = {}
    precisions = {}
    
    IoUs_list = []
    accuracies_list = []
    precisions_list = []
    
    for image_name in tqdm(renders.keys(), desc="Language embedding metric evaluation progress"):
        
        image_ious = {}
        image_accs = {}
        image_precs = {}
        
        image_ious_list = []
        image_accs_list = []
        image_precs_list = []
        
        for text in renders[image_name].keys():
            render = renders[image_name][text]
            gt = gts[image_name][text]

            image_ious[text] = mean_iou(render, gt).item()
            image_accs[text] = accuracy(render, gt).item()
            image_precs[text] = precision(render, gt).item()
            
            image_ious_list.append(image_ious[text])
            image_accs_list.append(image_accs[text])
            image_precs_list.append(image_precs[text])
        
        IoUs[image_name] = image_ious
        accuracies[image_name] = image_accs
        precisions[image_name] = image_precs
        
        IoUs_list.append(np.mean(image_ious_list))
        accuracies_list.append(np.mean(image_accs_list))
        precisions_list.append(np.mean(image_precs_list))
    
    print("  mIoU : {:>12.7f}".format(np.mean(IoUs_list), ".5"))
    print("  accuracy : {:>12.7f}".format(np.mean(accuracies_list), ".5"))
    print("  precision : {:>12.7f}".format(np.mean(precisions_list), ".5"))
    
    if json_pth:
        with open(json_pth, "w") as f:
            json.dump({"IoUs": IoUs, "accuracies": accuracies, "precisions": precisions}, f, indent=4)

if __name__ == "__main__":
    parser = configargparse.ArgParser(description="Training script parameters")
    parser.add_argument('--pred_path', '-pr', type=str, default="")
    parser.add_argument('--gt_path', '-gt', type=str, default="")
    parser.add_argument('--class_info','-js', type=str, required=True, help="Path to the JSON file containing text and class IDs")
    args = parser.parse_args()
    

    
    # Read class information
    texts, class_ids = read_class_info(args.class_info)
    
    # Lem
    lem_renders_dir = Path(args.pred_path)
    lem_gt_dir = Path(args.gt_path)

    renders, gts, names = read_3dovs_masks(lem_renders_dir / f"pred_segs", lem_gt_dir, (texts, class_ids))
    
    lem_evaluate(renders, gts, lem_renders_dir / "lem_metrics.json")
    mAP_evaluate(texts, class_ids, lem_renders_dir / "relevancy", lem_gt_dir, lem_renders_dir / "mAP_metrics.json")

    # example usage:
    # python3 eval_mod.py -pr Qwen/Qwen2.5-0.5B-Instruct_scene_021_30_forward -gt data/wayvescene/scene_021_30_forward/output_segformer -js Qwen/Qwen2.5-0.5B-Instruct_scene_021_30_forward.json 