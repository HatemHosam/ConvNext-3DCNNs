import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import convnext_tiny
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model

class Block3D(nn.Module):
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        # Adjusted dwconv layer to 3D
        self.dwconv = nn.Conv3d(dim, dim, kernel_size=(7, 7, 7), padding=(3, 3, 3), groups=dim)
        self.norm = LayerNorm3D(dim)
        self.pwconv1 = nn.Conv3d(dim, 4 * dim, kernel_size=1) 
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv3d(4 * dim, dim, kernel_size=1)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                   requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 4, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 4, 1, 2, 3) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x

class ConvNeXt3D(nn.Module):
    def __init__(self, in_chans=3, num_classes=1000, 
                 depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], drop_path_rate=0., 
                 layer_scale_init_value=1e-6, head_init_scale=1.,
                 ):
        super().__init__()

        self.downsample_layers = nn.ModuleList() 
        stem = nn.Sequential(
            nn.Conv3d(in_chans, dims[0], kernel_size=(4, 4, 4), stride=(4, 4, 4)),
            LayerNorm3D(dims[0])
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            # Adjusted downsample layer to 3D with kernel size (2, 2, 2) and stride (2, 2, 2)
            downsample_layer = nn.Sequential(
                LayerNorm3D(dims[i]),
                nn.Conv3d(dims[i], dims[i+1], kernel_size=(2, 2, 2), stride=(2, 2, 2)),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList() 
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block3D(dim=dims[i], drop_path=dp_rates[cur + j], 
                           layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1]) 
        self.head = nn.Linear(dims[-1], num_classes)

        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv3d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return self.norm(x.mean([-2, -1])) 

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x

class LayerNorm3D(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
    
    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x
		
def copy_weights_2d_to_3d(C2D_model, C3D_model):
    """
    Copy weights from pre-trained 2D ConvNext model to the modified 3D ConvNext model.

    Args:
        C2D_model (nn.Module): Pre-trained source model with 2D architecture.
        C3D_model (nn.Module): Destination model with modified 3D architecture.
    """
    # Copy stem weights
    C3D_model.downsample_layers[0][0].weight.data = C2D_model.downsample_layers[0][0].weight.data.unsqueeze(2).repeat(1, 1, C3D_model.downsample_layers[0][0].weight.shape[2], 1, 1)
    C3D_model.downsample_layers[0][0].bias.data = C2D_model.downsample_layers[0][0].bias.data

    for i in range(3):
        # Copy downsample layer weights
        C3D_model.downsample_layers[i][0][1].weight.data = C2D_model.downsample_layers[i][0][1].weight.data.unsqueeze(2).repeat(1, 1, C3D_model.downsample_layers[i][0][1].weight.shape[2], 1, 1)
        C3D_model.downsample_layers[i][0][1].bias.data = C2D_model.downsample_layers[i][0][1].bias.data

    # Copy block weights
    for i in range(4):
        for j in range(len(C3D_model.stages[i])):
            C3D_model.stages[i][j].dwconv.weight.data[:, :, 3, :, :] = C2D_model.stages[i][j].dwconv.weight.data
            C3D_model.stages[i][j].dwconv.bias.data = C2D_model.stages[i][j].dwconv.bias.data

            C3D_model.stages[i][j].pwconv1.weight.data = C2D_model.stages[i][j].pwconv1.weight.data.unsqueeze(2).repeat(1, 1, C3D_model.stages[i][j].pwconv1.weight.shape[2], 1, 1)
            C3D_model.stages[i][j].pwconv1.bias.data = C2D_model.stages[i][j].pwconv1.bias.data

            C3D_model.stages[i][j].pwconv2.weight.data = C2D_model.stages[i][j].pwconv2.weight.data.unsqueeze(2).repeat(1, 1, C3D_model.stages[i][j].pwconv2.weight.shape[2], 1, 1)
            C3D_model.stages[i][j].pwconv2.bias.data = C2D_model.stages[i][j].pwconv2.bias.data

    # Copy head weights
    C3D_model.head.weight.data = C2D_model.head.weight.data
    C3D_model.head.bias.data = C2D_model.head.bias.data
    
ConvNext_2D = convnext_tiny(weights='DEFAULT')
ConvNext3D_temp = ConvNeXt3D()

Inflated_ConvNext3D = copy_weights_2d_to_3d(ConvNext_2D, ConvNext3D_temp)
