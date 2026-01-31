from collections import OrderedDict
import io
import json
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List

from util.misc import NestedTensor, is_main_process
from .position_encoding import build_position_encoding

######### DINOv2 (ViT) backbone ##########
try:
    import timm
    _has_timm = True
except Exception:
    _has_timm = False
    
class VitBackboneDINO(nn.Module):
    """
    ViT(DINOv2) backbone wrapper
    """
    def __init__(self, model_name: str = "vit_base_patch14_dinov2",
                 train_backbone: bool = False,
                 return_interm_layers: bool = False):
        super().__init__()
        assert _has_timm, "need timm. recommend pip install timm>=0.9 "
        
        self._train_backbone = train_backbone
        self.model = timm.create_model(model_name, pretrained=True, num_classes=0, global_pool='') # generate timm model

        embed_dim = getattr(self.model, 'embed_dim', None)
        assert embed_dim is not None, f"timm {model_name}: failed to extract embed_dim"
        self.num_channels = embed_dim

        # 2) train / freeze 
        if train_backbone : 
            print(' Train ViT backbone')
        else : 
            print(' ViT backbone freezed')
            
        for name, p in self.model.named_parameters():
            p.requires_grad_(train_backbone)

        # 3) patch size & grid utils
        pe = getattr(self.model, 'patch_embed', None)
        assert pe is not None, "need ViT patch_embed."
        self.patch_size = pe.patch_size if isinstance(pe.patch_size, int) else pe.patch_size[0]

        self.return_interm_layers = return_interm_layers
        
        # (Optional) save VRAM with grad checkpointing 
        if hasattr(self.model, "set_grad_checkpointing"):
            print(getattr(self.model.blocks[0], "use_checkpoint", None))
            self.model.set_grad_checkpointing(True)
        

    @torch.no_grad()
    def _forward_features_no_grad(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model.forward_features(x)
        if isinstance(out, dict):
            for k in ['x_norm_patchtokens', 'x_norm', 'last_feat', 'tokens']:
                if k in out and out[k] is not None:
                    patches = out[k]  # (B, N, C) or (B, 1+N, C)
                    break
            else:
                raise RuntimeError("cannot find patch token key from timm forward_features dict.")
        else:
            patches = out  # (B, 1+N, C) or (B, N, C)

        # cls 토큰 제거
        if patches.dim() == 3 and patches.shape[1] >= 1 + 1:
            B, T, C = patches.shape
            return patches[:, 1:, :] if T != int(math.prod(self._guess_hw(patches, self.patch_size))) else patches
        return patches

    def _guess_hw(self, patches: torch.Tensor, patch: int):

        N = patches.shape[1] - 1
        h = int(math.sqrt(N))
        w = N // h
        return h, w

    def forward(self, tensor_list: NestedTensor) -> Dict[str, NestedTensor]:
        self.model.train(self._train_backbone)

        x, in_mask = tensor_list.tensors, tensor_list.mask  # x: (B,3,H,W), mask: (B,H,W)
        B, _, H, W = x.shape
       

        with torch.set_grad_enabled(any(p.requires_grad for p in self.model.parameters())):
            # dict or tensor
            out_tokens = self.model.forward_features(x)  # [B,3,518,518] -> [[B, 1370, 768]]

        # 정규화된 patch tokens 획득
        if isinstance(out_tokens, dict):
            patches = out_tokens.get('x_norm_patchtokens', None)
            if patches is None:
                patches = out_tokens.get('x_norm', None)
            if patches is None:

                patches = out_tokens.get('tokens', None)
            if patches is None:

                patches = out_tokens if torch.is_tensor(out_tokens) else None
        else:
            patches = out_tokens

        if patches is None:
            raise RuntimeError("could not found patch tokens in ViT forward_features")

        # cls 제거
        if patches.dim() == 3 and patches.shape[1] >= 1 + 1:
            # N = (Hp*Wp)
            patches = patches[:, 1:, :]

        B, N, C = patches.shape
        Hp = H // self.patch_size
        Wp = W // self.patch_size
        if Hp * Wp != N:
            # timm 내부 보간으로 인한 오차 방지
            grid = getattr(self.model.patch_embed, 'grid_size', None)
            if grid is not None:
                Hp, Wp = int(grid[0]), int(grid[1])
            else:
                Hp = int(math.sqrt(N))
                Wp = N // Hp
                
        # [B, N, Hp * Wp]
        feat = patches.permute(0, 2, 1).contiguous().view(B, C, Hp, Wp)  # [B, N, Hp * Wp] -> (B,C,Hp,Wp)

        # mask를 patch grid로 다운샘플
        assert in_mask is not None
        
        #  mask = F.interpolate(in_mask[None].float(), size=(Hp, Wp), mode='bilinear', align_corners=False).to(torch.bool)[0]

        mask = F.interpolate(   # in_mask: (B, H, W)  => (B, 1, H, W)로 채널 차원 추가
            in_mask.unsqueeze(1).float(),  # (B,1,H,W)
            size=(Hp, Wp),
            mode='nearest'                 
        ).squeeze(1).to(torch.bool)        # (B,Hp,Wp)

        return OrderedDict([("0", NestedTensor(feat, mask))])

# # -------------------------------------------------------------------------
# # (optional) torch.hub 경로: 공식 dinov2
# class VitBackboneDINOHub(nn.Module):
#     """
#     torch.hub: facebookresearch/dinov2
#     """
#     def __init__(self, hub_name: str = "dinov2_vitb14",
#                  train_backbone: bool = True,
#                  return_interm_layers: bool = False):
#         super().__init__()
#         self.training = train_backbone
#         self.model = torch.hub.load("facebookresearch/dinov2", hub_name)  # requires internet & repo

#         # embed_dim 추출
#         self.num_channels = getattr(self.model, 'embed_dim', 768)
#         # 동결/학습
#         for n, p in self.model.named_parameters():
#             p.requires_grad_(train_backbone)
#         # patch size
#         pe = getattr(self.model, 'patch_embed', None)
#         self.patch_size = pe.patch_size if pe is not None else 14
#         self.return_interm_layers = return_interm_layers

#     def forward(self, tensor_list: NestedTensor) -> Dict[str, NestedTensor]:
#         self.model.train(self.training)
#         x, in_mask = tensor_list.tensors, tensor_list.mask
#         B, _, H, W = x.shape
#         with torch.set_grad_enabled(any(p.requires_grad for p in self.model.parameters())):
#             # 공식 dinov2는 forward로 (B, C, Hp, Wp) 형태의 feature를 쉽게 얻기 어렵기 때문에
#             # 아래는 예시: get_intermediate_layers로 토큰 받고 재배열
#             tokens = self.model.get_intermediate_layers(x, n=1, reshape=False)[0]  # (B, 1+N, C)
#         patches = tokens[:, 1:, :]
#         B, N, C = patches.shape
#         Hp = H // self.patch_size
#         Wp = W // self.patch_size
#         if Hp * Wp != N:
#             Hp = int(math.sqrt(N)); Wp = N // Hp
#         feat = patches.permute(0, 2, 1).contiguous().view(B, C, Hp, Wp)
#         # mask = F.interpolate(in_mask[None].float(), size=(Hp, Wp), mode='bilinear', align_corners=False).to(torch.bool)[0]
#         mask = F.interpolate(in_mask[0:1].float(), size=(Hp, Wp), mode='nearest')[0].to(torch.bool)
#         return OrderedDict([("0", NestedTensor(feat, mask))])


######### original ResNet backbone ##########

class FrozenBatchNorm2d(torch.nn.Module):
    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class BackboneBase(nn.Module):

    def __init__(self, backbone: nn.Module, train_backbone: bool, num_channels: int, return_interm_layers: bool):
        super().__init__()
        for name, parameter in backbone.named_parameters():
            if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)
        if return_interm_layers:
            return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
        else:
            return_layers = {'layer4': "0"}
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.num_channels = num_channels

    def forward(self, tensor_list: NestedTensor):
        xs = self.body(tensor_list.tensors)
        out: Dict[str, NestedTensor] = {}
        for name, x in xs.items():
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(x, mask)
        return out


class Backbone(BackboneBase):
    def __init__(self, name: str,
                 train_backbone: bool,
                 return_interm_layers: bool,
                 dilation: bool):
        backbone = getattr(torchvision.models, name)(
            replace_stride_with_dilation=[False, False, dilation],
            pretrained=is_main_process(), norm_layer=FrozenBatchNorm2d)
        num_channels = 512 if name in ('resnet18', 'resnet34') else 2048
        super().__init__(backbone, train_backbone, num_channels, return_interm_layers)


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)
        out: List[NestedTensor] = []
        pos = []
        for name, x in xs.items():
            out.append(x)
            pos.append(self[1](x).to(x.tensors.dtype))
        return out, pos


def build_backbone(args):
    position_embedding = build_position_encoding(args)    
    train_backbone = args.lr_backbone > 0
    return_interm_layers = args.masks
    
    name = args.backbone.lower()
    print(' backbone :', name)
    
    # 선택 분기: DINOv2 (timm / torch.hub) vs ResNet

    if name.startswith("vit") and "dinov2" in name:
        vit = VitBackboneDINO(model_name=name, train_backbone=train_backbone,
                              return_interm_layers=return_interm_layers)
        model = Joiner(vit, position_embedding)
        model.num_channels = vit.num_channels
        return model
    if name.startswith("dinov2_"):  # 예: dinov2_vitb14 (torch.hub)
        vit = VitBackboneDINOHub(hub_name=name, train_backbone=train_backbone,
                                 return_interm_layers=return_interm_layers)
        model = Joiner(vit, position_embedding)
        model.num_channels = vit.num_channels
        return model
    # import pdb;pdb.set_trace()
    # 기존 ResNet
    backbone = Backbone(args.backbone, train_backbone, return_interm_layers, args.dilation)
    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels
    return model
