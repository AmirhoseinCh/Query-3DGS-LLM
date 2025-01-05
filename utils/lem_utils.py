import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
from torchvision import transforms
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer

from PIL import Image
import colorsys

import open_clip

def pca(features):
    """
    Perform PCA on the given features and return the result.
    Args:
        features: (C, H, W) torch.Tensor
    Returns:
        (3, H, W) torch.Tensor
    """
    shape = features.shape
    
    np_features = features.permute(1,2,0).reshape(-1, shape[0]).cpu().numpy()

    # Handle NaN values by imputing with the mean
    imputer = SimpleImputer(strategy='mean')
    np_features = imputer.fit_transform(np_features)

    pca = PCA(n_components=3)
    pca.fit(np_features)

    pca_features = pca.transform(np_features)
    pca_features = (pca_features - pca_features.min()) / (pca_features.max() - pca_features.min())

    pca_features = torch.from_numpy(pca_features).reshape(shape[1], shape[2], 3).permute(2, 0, 1) 
    
    return pca_features

def index_to_rgb_images(input_tensor, color_map):
    """
    Args:
        input_tensor (torch.Tensor): (B, H, W, 1)
        color_map (torch.Tensor): (N, 3)
    Returns:
        _type_: (B, H, W, 3)
    """
    index_tensor = input_tensor[:, :, :, 0].long()
    rgb_images = color_map[index_tensor]

    return rgb_images

def generate_colors(n):
    colors = []
    for i in range(n):
        hue = i / float(n)  # 在色轮上均匀分布的色调值
        rgb = colorsys.hsv_to_rgb(hue, 1.0, 1.0)  # 将 HSV 色彩空间转换为 RGB
        color = torch.tensor(rgb)  # RGB 值在 [0, 1]
        colors.append(color)
    return torch.stack(colors, dim=0)

def read_codebook(path):
    return torch.load(path)['embedding.weight']

def index_to_featrues(indies_tensor, codebook):
    """
    Args:
        input_tensor (torch.Tensor): (B, H, W, 1)
        color_map (torch.Tensor): (N, x)
    Returns:
        _type_: (B, H, W, x)
    """
    index_tensor = indies_tensor[:, :, :, 0].long()
    features = codebook[index_tensor]

    return features

class CLIPRelevance:
    def __init__(self, device="cuda"):
        self.device = device
        model, _, _ = open_clip.create_model_and_transforms('ViT-B-16', pretrained='laion2b_s34b_b88k')
        model.eval()
        self.model = model.to(self.device)
        self.tokenizer = open_clip.get_tokenizer('ViT-B-16')
        self.process = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Resize((224, 224)),
                    transforms.Normalize(
                        mean=[0.48145466, 0.4578275, 0.40821073],
                        std=[0.26862954, 0.26130258, 0.27577711],
                    ),
                ]
            )

        # self.negatives = negatives 
        # self.positives = positives


        # self.negatives = ("object", "things", "stuff", "texture")#("Trees", "Pedestrian", "Buildings")#("Buildings","Vehicle", "Pedsetrian", "Trees")#("traffic", "vehicle", "traffic light", "Buildings")#("traffic", "vehicle", "pedsetrian","ground")#("object", "things", "stuff", "texture")
        # self.positives = ("Signage", "Traffic sign", "Stop")#("Traffic", "Moving vehicle", "Truck", "Left")#("Signage", "Traffic sign", "Information board")#("Crosswalk", "Person")#("traffic light", "stoplight", "traffic signal")
        # with torch.no_grad():
        #     if self.negatives:
        #         tok_phrases = torch.cat([self.tokenizer(phrase) for phrase in self.negatives]).to(self.device)
        #         self.neg_embeds = self.model.encode_text(tok_phrases)
        #         self.neg_embeds /= self.neg_embeds.norm(dim=-1, keepdim=True)
        #     else:
        #         self.neg_embeds = None
            

        #     if self.positives:
        #         tok_pos_phrases = torch.cat([self.tokenizer(phrase) for phrase in self.positives]).to(self.device)
        #         self.pos_embeds = self.model.encode_text(tok_pos_phrases)
        #         self.pos_embeds /= self.pos_embeds.norm(dim=-1, keepdim=True)
        #     else:
        #         self.pos_embeds = None

    def encode_text(self, texts):
        with torch.no_grad():
            tok_phrases = torch.cat([self.tokenizer(text) for text in texts]).to(self.device)
            embeds = self.model.encode_text(tok_phrases)
        embeds /= embeds.norm(dim=-1, keepdim=True)
        return embeds
    
    def encode_image(self, image):
        with torch.no_grad():
            embeds = self.model.encode_image(self.process(image).to(self.device)[None, ...])
        embeds /= embeds.norm(dim=-1, keepdim=True)
        return embeds
    """
    def get_relevancy(self, embed: torch.Tensor, positive: str or Image, negatives=None, scale = 100) -> torch.Tensor:
        if isinstance(positive, str):
            # pos_embeds = self.encode_text([f"a photo of a {positive}"])
            pos_embeds = self.encode_text([f"{positive}"])
        else:
            pos_embeds = self.encode_image(positive)
        
        if self.pos_embeds is not None:
            pos_embeds = torch.cat([pos_embeds, self.pos_embeds], dim=0)

        
        if negatives is not None:
            with torch.no_grad():
                tok_phrases = torch.cat([self.tokenizer(v) for v in negatives.items()]).to(self.device)
                out_neg_embeds = self.model.encode_text(tok_phrases)
            out_neg_embeds /= out_neg_embeds.norm(dim=-1, keepdim=True)
            if self.neg_embeds is not None:
                phrases_embeds = torch.cat([pos_embeds, self.neg_embeds, out_neg_embeds], dim=0)
            else:
                phrases_embeds = torch.cat([pos_embeds, out_neg_embeds], dim=0)
        elif self.neg_embeds is not None:
            phrases_embeds = torch.cat([pos_embeds, self.neg_embeds], dim=0)
        else:
            phrases_embeds = pos_embeds


        
        p = phrases_embeds.to(embed.dtype)  # phrases x 512



        # output = torch.matmul(embed, p.T)  # hw x phrases
        # output = F.cosine_similarity(embed[..., None, :], p[None, None, ...], dim=-1)  # hw x phrases
        output = self._cosine_sim(embed, p)

        # print("---output:",output.shape)
        
        num_positives = len(self.positives) + 1  # +1 for the input positive
        positive_vals = output[..., :num_positives]  # hw x num_positives
        negative_vals = output[..., num_positives:]  # hw x N_negative
        # Use the maximum similarity among positive embeddings
        max_positive_vals = positive_vals.max(dim=-1, keepdim=True)[0]  # hw x 1
        repeated_pos = max_positive_vals.repeat(1, 1, len(self.negatives))  # hw x N_negative

        sims = torch.stack((repeated_pos, negative_vals), dim=-1)  # hw x N-phrase x 2

        # print("---sims:",sims.shape)

        softmax = torch.softmax(scale*sims, dim=-1)  # hw x n-phrase x 2
        best_id = softmax[..., 0].argmin(dim=2)  # hw
        return torch.gather(softmax, 2, best_id[..., None, None].expand(best_id.shape[0], best_id.shape[1], len(self.negatives), 2))[
            :, :, 0, :
        ]
    """

    def get_relevancy(self, embed: torch.Tensor, main_positive: str, helping_positives: list = None, 
                        negatives: list = None, scale = 100) -> torch.Tensor:
            # Encode main positive
            main_pos_embed = self.encode_text([main_positive])
            
            # Encode helping positives if provided
            if helping_positives:
                helping_pos_embeds = self.encode_text(helping_positives)
                pos_embeds = torch.cat([main_pos_embed, helping_pos_embeds], dim=0)
            else:
                pos_embeds = main_pos_embed
            
            # Encode negatives if provided
            if negatives:
                neg_embeds = self.encode_text(negatives)
                phrases_embeds = torch.cat([pos_embeds, neg_embeds], dim=0)
            else:
                # If no negatives, create a dummy negative
                dummy_neg = self.encode_text(["object", "things", "stuff", "texture"])
                phrases_embeds = torch.cat([pos_embeds, dummy_neg], dim=0)

            p = phrases_embeds.to(embed.dtype)

            output = self._cosine_sim(embed, p)
            
            num_positives = pos_embeds.shape[0]
            num_negatives = phrases_embeds.shape[0] - num_positives
            
            positive_vals = output[..., :num_positives]  # hw x num_positives
            negative_vals = output[..., num_positives:]  # hw x N_negative
            
            
            # If we have fewer positives than negatives, append the main positive similarity
            if num_positives < num_negatives:
                main_positive_sim = output[..., 0:1]  # This is the main positive cosine similarity
                repeats_needed = num_negatives - num_positives
                repeated_main_positive = main_positive_sim.repeat(1, 1, repeats_needed)
                positive_vals = torch.cat([positive_vals, repeated_main_positive], dim=-1)
            elif num_positives > num_negatives:
                repeats_needed = num_positives - num_negatives
                main_negative_sim = output[..., num_positives:num_positives+1]  
                repeated_main_negative = main_negative_sim.repeat(1, 1, repeats_needed)
                negative_vals = torch.cat([negative_vals, repeated_main_negative], dim=-1)
                
            sims = torch.stack((positive_vals, negative_vals), dim=-1)  # hw x N-negative x 2
            softmax = torch.softmax(scale*sims, dim=-1)  # hw x n-negative x 2
            best_id = softmax[..., 0].argmin(dim=2)  # hw
            
            score = torch.gather(softmax, 2, best_id[..., None, None].expand(best_id.shape[0], best_id.shape[1], negative_vals.shape[-1], 2))[
                :, :, 0, :
            ]
            # print('--output:',output.shape) #torch.Size([540, 960, 4])
            # print('--sim:', sims.shape) #torch.Size([540, 960, 3, 2])
            # print('--softmax:',softmax.shape) #torch.Size([540, 960, 3, 2])
            # print('--best_id:',best_id.shape) #torch.Size([540, 960])
            # print('--score:', score.shape) #torch.Size([540, 960, 2])
            return score

    def get_simlarity(self, embed: torch.Tensor, positive: str or Image) -> torch.Tensor:
        if isinstance(positive, str):
            pos_embeds = self.encode_text([positive])
        else:
            pos_embeds = self.encode_image(positive)
        sim = F.cosine_similarity(embed, pos_embeds[None, ...], dim=-1)
        return sim
    
    def _cosine_sim(self, a, b):
        a_norm = torch.norm(a, dim=-1)[...,None]
        b_norm = torch.norm(b, dim=-1)[None,...]
        d = torch.matmul(a, b.t()) / (torch.matmul(a_norm, b_norm) + 1e-6)
        return d

class Embedding(nn.Module):
    def __init__(self, in_channels, N_freqs, logscale=True):
        """
        Defines a function that embeds x to (x, sin(2^k x), cos(2^k x), ...)
        in_channels: number of input channels (3 for both xyz and direction)
        """
        super(Embedding, self).__init__()
        self.N_freqs = N_freqs
        self.in_channels = in_channels
        self.funcs = [torch.sin, torch.cos]
        self.out_channels = in_channels*(len(self.funcs)*N_freqs+1)

        if logscale:
            self.freq_bands = 2**torch.linspace(0, N_freqs-1, N_freqs)
        else:
            self.freq_bands = torch.linspace(1, 2**(N_freqs-1), N_freqs)

    def forward(self, x):
        """
        Embeds x to (x, sin(2^k x), cos(2^k x), ...) 
        Different from the paper, "x" is also in the output
        See https://github.com/bmild/nerf/issues/12

        Inputs:
            x: (B, self.in_channels)

        Outputs:
            out: (B, self.out_channels)
        """
        out = [x]
        for freq in self.freq_bands:
            for func in self.funcs:
                out += [func(freq*x)]

        return torch.cat(out, -1)