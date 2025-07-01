from transformers import DepthAnythingForDepthEstimation
import torch
from torch import nn
from torch.nn.parameter import Parameter
import math
 
# Simplified RaSA block
class RaSA_qkv(nn.Module):
    def __init__(self, base_linear, shared_rank_pool, rank_weights):
        super().__init__()
        self.base = base_linear
        self.shared_pool = shared_rank_pool  # [dim, r]
        self.rank_weights = rank_weights     # [r, dim]
 
    def forward(self, x):
        base_out = self.base(x)
        delta = x @ self.shared_pool @ self.rank_weights
        return base_out + delta
 
class DepthAnythingDepthEstimationHead(nn.Module):
    def __init__(self, model_head):
        super().__init__()
        self.conv1 = model_head.conv1
        self.conv2 = model_head.conv2
        self.activation1 = nn.ReLU()
        self.conv3 = model_head.conv3
        self.activation2 = nn.Sigmoid()
 
    def forward(self, hidden_states, height, width):
        x = self.conv1(hidden_states)
        x = nn.functional.interpolate(x, (int(height), int(width)), mode="bilinear", align_corners=True)
        x = self.activation1(self.conv2(x))
        return self.activation2(self.conv3(x))
 
class RaSAInitializer:
    def __init__(self, model, shared_rank=16, replace_modules=['q', 'v']):
        self.model = model
        self.shared_rank = shared_rank
        self.replace_modules = replace_modules
        self.shared_pool = None
        self.rank_weights = []
        self.initialize_rasa()
 
    def initialize_rasa(self):
        dim = self.model.backbone.encoder.layer[0].attention.attention.query.in_features
        self.shared_pool = nn.Parameter(torch.randn(dim, self.shared_rank))
 
        for blk in self.model.backbone.encoder.layer:
            for t in self.replace_modules:
                base_layer = getattr(blk.attention.attention, {'q': 'query', 'k': 'key', 'v': 'value'}[t])
                rank_weight = nn.Parameter(torch.randn(self.shared_rank, dim))
                wrapped = RaSA_qkv(base_layer, self.shared_pool, rank_weight)
                setattr(blk.attention.attention, {'q': 'query', 'k': 'key', 'v': 'value'}[t], wrapped)
                self.rank_weights.append(rank_weight)
 
        print("Replaced LoRA with RaSA!")
 
class DARES(nn.Module):
    def __init__(self, shared_rank=16, replace_modules=['q', 'v']):
        super().__init__()
        model = DepthAnythingForDepthEstimation.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf")
        self.config = model.config
        self.backbone = model.backbone
 
        # Apply RaSA
        self.rasa_initializer = RaSAInitializer(model, shared_rank, replace_modules)
 
        self.neck = model.neck
        self.head = DepthAnythingDepthEstimationHead(model.head)
        model.post_init()
 
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
 
        return {
            ("disp", 0): self.head(hidden_states[3], height, width),
            ("disp", 1): self.head(hidden_states[2], height / 2, width / 2),
            ("disp", 2): self.head(hidden_states[1], height / 4, width / 4),
            ("disp", 3): self.head(hidden_states[0], height / 8, width / 8),
        }
