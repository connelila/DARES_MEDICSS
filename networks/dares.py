from transformers import AutoImageProcessor, AutoModelForDepthEstimation, DepthAnythingForDepthEstimation
import torch
from torchvision import transforms
import numpy as np
from PIL import Image
import requests
import matplotlib.pyplot as plt
import os
import torch.nn as nn
import math
import torch.nn.functional as F
from torch.nn.parameter import Parameter

class _Rasa_qkv(nn.Module):
    def __init__(self, w: nn.Module, r: int, rasa_alpha=1.0, rasa_dropout=0.0, rasa_k=0):
        super().__init__()
        self.w = w
        self.dim = w.in_features
        self.r = r
        self.rasa_alpha = rasa_alpha
        self.rasa_dropout = nn.Dropout(p=rasa_dropout) if rasa_dropout > 0 else nn.Identity()
        self.rasa_k = rasa_k

        # Effective rank after considering rasa_k
        effective_r = max(1, r - rasa_k)  # Ensure at least rank 1
        
        self.rasa_A = nn.Linear(self.dim, effective_r, bias=False)
        self.rasa_B = nn.Linear(effective_r, self.w.out_features, bias=False)

        # Mixing weights (learnable parameter)
        self.mixing_weight = nn.Parameter(torch.tensor(0.5 * rasa_alpha))

        # Initialize parameters
        nn.init.kaiming_uniform_(self.rasa_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.rasa_B.weight)

    def forward(self, x):
        W = self.w(x)  # base output
        delta = self.rasa_B(self.rasa_dropout(self.rasa_A(x)))
        # Apply mixing weight with proper scaling
        W = W + delta * self.mixing_weight
        return W

class DepthAnythingDepthEstimationHead(nn.Module):
    def __init__(self, model_head):
        super().__init__()
        self.conv1 = model_head.conv1
        self.conv2 = model_head.conv2
        self.activation1 = nn.ReLU()
        self.conv3 = model_head.conv3
        self.activation2 = nn.Sigmoid()

    def forward(self, hidden_states, height, width):
        predicted_depth = self.conv1(hidden_states)
        predicted_depth = nn.functional.interpolate(
            predicted_depth,
            (int(height), int(width)),
            mode="bilinear",
            align_corners=True,
        )
        predicted_depth = self.conv2(predicted_depth)
        predicted_depth = self.activation1(predicted_depth)
        predicted_depth = self.conv3(predicted_depth)
        predicted_depth = self.activation2(predicted_depth)
        return predicted_depth

class RasaInitializer:
    def __init__(self, model, r=[14,14,12,12,10,10,8,8,8,8,8,8], lora=['q', 'v'], rasa_alpha=1.0, rasa_dropout=0.0, rasa_k=0):
        self.model = model
        self.r = r
        self.lora = lora
        self.rasa_alpha = rasa_alpha
        self.rasa_dropout = rasa_dropout
        self.rasa_k = rasa_k
        self.initialize_rasa()

    def initialize_rasa(self):
        # Freeze backbone parameters
        for param in self.model.backbone.parameters():
            param.requires_grad = False

        for t_layer_i, blk in enumerate(self.model.backbone.encoder.layer):
            dim = blk.attention.attention.query.in_features

            if 'q' in self.lora:
                w_q = blk.attention.attention.query
                blk.attention.attention.query = _Rasa_qkv(
                    w_q,
                    r=self.r[t_layer_i],
                    rasa_alpha=self.rasa_alpha,
                    rasa_dropout=self.rasa_dropout,
                    rasa_k=self.rasa_k,
                )

            if 'v' in self.lora:
                w_v = blk.attention.attention.value
                blk.attention.attention.value = _Rasa_qkv(
                    w_v,
                    r=self.r[t_layer_i],
                    rasa_alpha=self.rasa_alpha,
                    rasa_dropout=self.rasa_dropout,
                    rasa_k=self.rasa_k,
                )

            if 'k' in self.lora:
                w_k = blk.attention.attention.key
                blk.attention.attention.key = _Rasa_qkv(
                    w_k,
                    r=self.r[t_layer_i],
                    rasa_alpha=self.rasa_alpha,
                    rasa_dropout=self.rasa_dropout,
                    rasa_k=self.rasa_k,
                )

        print("RaSA adapters initialized!")

class DARES(nn.Module):
    def __init__(self, r=[14,14,12,12,10,10,8,8,8,8,8,8], lora=['q', 'v'], rasa_alpha=1.0, rasa_dropout=0.0, rasa_k=0):
        super(DARES, self).__init__()
        model = DepthAnythingForDepthEstimation.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf")
        self.r = r
        self.lora = lora
        self.config = model.config
        self.backbone = model.backbone

        # Initialize RaSA parameters
        self.rasa_initializer = RasaInitializer(
            model,
            r=r,
            lora=lora,
            rasa_alpha=rasa_alpha,
            rasa_dropout=rasa_dropout,
            rasa_k=rasa_k
        )

        self.neck = model.neck
        model_head = model.head
        self.head = DepthAnythingDepthEstimationHead(model_head)
        model.post_init()

    def save_parameters(self, filename: str) -> None:
        assert filename.endswith(".pt") or filename.endswith('.pth')

        # Collect RaSA parameters
        rasa_A_tensors = {}
        rasa_B_tensors = {}
        mixing_weights = {}

        for t_layer_i, blk in enumerate(self.backbone.encoder.layer):
            if 'q' in self.lora:
                layer_q = blk.attention.attention.query
                if hasattr(layer_q, 'rasa_A') and hasattr(layer_q, 'rasa_B'):
                    rasa_A_tensors[f"rasa_A_q_{t_layer_i:03d}"] = layer_q.rasa_A.weight.data.clone()
                    rasa_B_tensors[f"rasa_B_q_{t_layer_i:03d}"] = layer_q.rasa_B.weight.data.clone()
                    mixing_weights[f"mixing_weight_q_{t_layer_i:03d}"] = layer_q.mixing_weight.data.clone()
            
            if 'v' in self.lora:
                layer_v = blk.attention.attention.value
                if hasattr(layer_v, 'rasa_A') and hasattr(layer_v, 'rasa_B'):
                    rasa_A_tensors[f"rasa_A_v_{t_layer_i:03d}"] = layer_v.rasa_A.weight.data.clone()
                    rasa_B_tensors[f"rasa_B_v_{t_layer_i:03d}"] = layer_v.rasa_B.weight.data.clone()
                    mixing_weights[f"mixing_weight_v_{t_layer_i:03d}"] = layer_v.mixing_weight.data.clone()
            
            if 'k' in self.lora:
                layer_k = blk.attention.attention.key
                if hasattr(layer_k, 'rasa_A') and hasattr(layer_k, 'rasa_B'):
                    rasa_A_tensors[f"rasa_A_k_{t_layer_i:03d}"] = layer_k.rasa_A.weight.data.clone()
                    rasa_B_tensors[f"rasa_B_k_{t_layer_i:03d}"] = layer_k.rasa_B.weight.data.clone()
                    mixing_weights[f"mixing_weight_k_{t_layer_i:03d}"] = layer_k.mixing_weight.data.clone()

        # Save head parameters
        decode_head_tensors = self.head.state_dict()

        merged_dict = {**rasa_A_tensors, **rasa_B_tensors, **mixing_weights, **decode_head_tensors}
        torch.save(merged_dict, filename)

        print(f"Saved RaSA parameters to {filename}.")

    def load_parameters(self, filename: str, device: str) -> None:
        assert filename.endswith(".pt") or filename.endswith('.pth')

        state_dict = torch.load(filename, map_location=device)

        # Load RaSA parameters back into layers
        for t_layer_i, blk in enumerate(self.backbone.encoder.layer):
            if 'q' in self.lora:
                layer_q = blk.attention.attention.query
                if hasattr(layer_q, 'rasa_A') and hasattr(layer_q, 'rasa_B'):
                    layer_q.rasa_A.weight.data = state_dict[f"rasa_A_q_{t_layer_i:03d}"].to(device)
                    layer_q.rasa_B.weight.data = state_dict[f"rasa_B_q_{t_layer_i:03d}"].to(device)
                    if f"mixing_weight_q_{t_layer_i:03d}" in state_dict:
                        layer_q.mixing_weight.data = state_dict[f"mixing_weight_q_{t_layer_i:03d}"].to(device)
            
            if 'v' in self.lora:
                layer_v = blk.attention.attention.value
                if hasattr(layer_v, 'rasa_A') and hasattr(layer_v, 'rasa_B'):
                    layer_v.rasa_A.weight.data = state_dict[f"rasa_A_v_{t_layer_i:03d}"].to(device)
                    layer_v.rasa_B.weight.data = state_dict[f"rasa_B_v_{t_layer_i:03d}"].to(device)
                    if f"mixing_weight_v_{t_layer_i:03d}" in state_dict:
                        layer_v.mixing_weight.data = state_dict[f"mixing_weight_v_{t_layer_i:03d}"].to(device)
            
            if 'k' in self.lora:
                layer_k = blk.attention.attention.key
                if hasattr(layer_k, 'rasa_A') and hasattr(layer_k, 'rasa_B'):
                    layer_k.rasa_A.weight.data = state_dict[f"rasa_A_k_{t_layer_i:03d}"].to(device)
                    layer_k.rasa_B.weight.data = state_dict[f"rasa_B_k_{t_layer_i:03d}"].to(device)
                    if f"mixing_weight_k_{t_layer_i:03d}" in state_dict:
                        layer_k.mixing_weight.data = state_dict[f"mixing_weight_k_{t_layer_i:03d}"].to(device)

        # Load head parameters
        decode_head_dict = self.head.state_dict()
        decode_head_keys = decode_head_dict.keys()
        decode_head_new_state_dict = {k: state_dict[k].to(device) for k in decode_head_keys if k in state_dict}
        decode_head_dict.update(decode_head_new_state_dict)
        self.head.load_state_dict(decode_head_dict)

        print(f"Loaded RaSA parameters from {filename}.")

    def forward(self, pixel_values):
        outputs = self.backbone.forward_with_filtered_kwargs(
            pixel_values, output_hidden_states=None, output_attentions=None
        )
        hidden_states = outputs.feature_maps
        _, _, height, width = pixel_values.shape
        patch_size = self.config.patch_size
        patch_height = height // patch_size
        patch_width = width // patch_size
        hidden_states = self.neck(hidden_states, patch_height, patch_width)
        outputs = {}
        outputs[("disp", 0)] = self.head(hidden_states[3], height, width)
        outputs[("disp", 1)] = self.head(hidden_states[2], height / 2, width / 2)
        outputs[("disp", 2)] = self.head(hidden_states[1], height / 4, width / 4)
        outputs[("disp", 3)] = self.head(hidden_states[0], height / 8, width / 8)
        return outputs
