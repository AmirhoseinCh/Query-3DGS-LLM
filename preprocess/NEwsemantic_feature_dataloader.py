import json
from tqdm import tqdm
from dataclasses import dataclass
from PIL import Image
from pathlib import Path

import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from torchvision import transforms

import open_clip

from clip.clip_utils import *
from clip.dense_extractor.dense_extractor import DenseCLIPExtractorParams, DenseCLIPExtractor
from clip.pyramid.mixture_embedding_dataloader import MixtureEmbeddingDataloader
from dino.dino_dataloader import DinoExtractorParams, get_dinos

class SematicFeatureDataset(Dataset):
    def __init__(self, path, clip_params: DenseCLIPExtractorParams = None, dino_params: DinoExtractorParams = None):
        super().__init__()
        self.path = path
        self.clip_params = clip_params if clip_params else DenseCLIPExtractorParams()
        self.dino_params = dino_params if dino_params else DinoExtractorParams()
        
        self._concat_features()
    
    def _concat_features(self):
        clips = DenseCLIPExtractor(self.path, self.clip_params).data.to("cuda") # [194, 512, 106, 160]

        # stride4: [N, 384, 55, 83]
        # stride2: [N, 384, 109, 146]
        dinos = get_dinos(self.path, self.dino_params, half=True).permute(0, 3, 1, 2).to("cuda")
        
        # upsample dino feature map to clip feature map size
        h, w = clips.shape[2], clips.shape[3]
        if dinos.shape[0] < 300:
            sampled_dinos = F.interpolate(dinos, size=(h, w), mode='bilinear', align_corners=False)
        else:
            # TO avoid RuntimeError: upsample_bilinear2d_nhwc only supports input tensors with less than INT_MAX elements
            sampled_dinos = torch.cat((F.interpolate(dinos[:200], size=(h, w), mode='bilinear', align_corners=False), 
                                       F.interpolate(dinos[200:], size=(h, w), mode='bilinear', align_corners=False)), 
                                      dim=0)
        
        del dinos
        torch.cuda.empty_cache()
        
        self.data = torch.cat((clips, sampled_dinos), dim=1)
        del clips, sampled_dinos
        
        self.data = self.data.contiguous().permute(0, 2, 3, 1)
        mask = torch.isnan(self.data) | torch.isinf(self.data)
        self.data[mask] = 0
        
    def _contract_feature(self):
        # read matches info from file
        with open("/media/nas/gsh/match_json/room_match.json", "r") as f:
            matches = json.loads(f.read())
        # contract
        for key in tqdm(matches.keys()):
            indices = torch.tensor(matches[key])
            n_indices = indices[:, 0].long()
            h_indices = indices[:, 1].long()
            w_indices = indices[:, 2].long()
            avg = self.data[n_indices, h_indices, w_indices, :512].mean(dim=0)
            self.data[n_indices, h_indices, w_indices, :512] = avg
        
    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        return self.data[index]
    
    
class PyramidSematicFeatureDataset(Dataset):
    def __init__(self, path, clip_params: DenseCLIPExtractorParams = None, dino_params: DinoExtractorParams = None):
        super().__init__()
        self.path = path
        self.dino_params = dino_params if dino_params else DinoExtractorParams()
        
        self.device = "cuda"
        model, _, _ = open_clip.create_model_and_transforms('ViT-B-16', pretrained='laion2b_s34b_b88k')
        model.eval()
        self.model = model.to(self.device)
        self.process = transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.Normalize(
                        mean=[0.48145466, 0.4578275, 0.40821073],
                        std=[0.26862954, 0.26130258, 0.27577711],
                    ),
                ]
            )
        
        self.image_paths = get_image_paths(path)
        self.image_shape = to_tensor(Image.open(self.image_paths[0])).shape[1:3]

        self.cfg = {
                "tile_size_range": [0.05, 0.5],
                "tile_size_res": 7,
                "stride_scaler": 0.5,
                "image_shape": list(self.image_shape),
                "model_name": "ViT-B/16",
            }
                
        self.mixture_path = Path(os.path.join(path, f"pyramid_{self.cfg['tile_size_range'][0]}", "cache"))

        self._concat_features()
    
    def _get_images(self):
        image_list = []
        for image_path in tqdm(self.image_paths):
            image_list.append(to_tensor(Image.open(image_path)))
        return torch.stack(image_list)
    
    def _load_mixture(self):
        mixture_data = None
        if self.mixture_path.with_suffix(".npy").exists():
            mixture_data = MixtureEmbeddingDataloader(
                device = self.device,
                cfg = self.cfg,
                cache_path = self.mixture_path,
                model = self.model,
                process = self.process,
            )
        else:
            image_list = self._get_images()
            mixture_data = MixtureEmbeddingDataloader(
                device = self.device,
                cfg = self.cfg,
                image_list = image_list,
                cache_path = self.mixture_path,
                model = self.model,
                process = self.process,
            )
        return mixture_data().to(self.device)

    def _concat_features(self):
        clips = self._load_mixture().permute(0, 3, 1, 2) # [194, 512, 106, 160]

        # stride4: [N, 384, 55, 83]
        # stride2: [N, 384, 109, 146]
        dinos = get_dinos(self.path, self.dino_params, half=True).permute(0, 3, 1, 2).to("cuda")
        print('clip shape:', clips.shape)
        print('dino shape:', dinos.shape)
        # upsample clip feature map to dino feature map size
        h, w = 60, 90
 

        # Function to handle interpolation in chunks
        def interpolate_in_chunks(tensor, size, chunk_size=100, mode='nearest'):
            result_cpu = []
            for i in range(0, tensor.shape[0], chunk_size):
                print(f"Interpolating chunk: {i}")
                chunk = tensor[i:i + chunk_size]
                chunk = F.interpolate(chunk, size=size, mode=mode)
                result_cpu.append(chunk.cpu())
                del chunk
                torch.cuda.empty_cache()
            return torch.cat(result_cpu, dim=0)
        # Upsample clip feature map to dino feature map size using chunked interpolation
        sampled_clips = interpolate_in_chunks(clips, size=(h, w), chunk_size=200, mode='nearest')
        
        # Free up GPU memory by deleting the clips tensor and clearing the cache
        del clips
        torch.cuda.empty_cache()
        
        # Interpolate and concatenate the dinos tensor in smaller chunks and offload to CPU
        sampled_dinos = interpolate_in_chunks(dinos, size=(h, w), chunk_size=200, mode='nearest')

        # Free up GPU memory by deleting the dinos tensor and clearing the cache
        del dinos
        torch.cuda.empty_cache()
        
        

        print('sampled clip shape:', sampled_clips.shape)
        print('sampled dino shape:', sampled_dinos.shape)
        
        # Concatenate the final results on CPU
        final_data_cpu = torch.cat((sampled_clips, sampled_dinos), dim=1)
        print('final shape:', final_data_cpu.shape)
        # Process in smaller segments if necessary
        def process_in_segments(tensor, segment_size=100):
            result_segments = []
            for i in range(0, tensor.shape[0], segment_size):
                # print(f"masking segments: {i}")
                segment = tensor[i:i + segment_size].to("cuda")
                segment = segment.contiguous().permute(0, 2, 3, 1)
                mask = torch.isnan(segment) | torch.isinf(segment)
                segment[mask] = 0
                result_segments.append(segment.cpu())
                del segment
                torch.cuda.empty_cache()
            return torch.cat(result_segments, dim=0).to("cuda")

        # Process the concatenated tensor in segments to avoid memory overflow
        self.data = process_in_segments(final_data_cpu, segment_size=100)
        print('final data:', self.data.device)

        del sampled_clips, sampled_dinos, final_data_cpu
        torch.cuda.empty_cache()
        

    

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        return self.data[index]