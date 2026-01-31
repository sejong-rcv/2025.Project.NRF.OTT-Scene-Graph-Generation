import torch.utils.data
import torchvision

from .ag_single import build as build_ag_single
from .ag_multi import build as build_ag_multi

from .ag_single_dinov2 import build as build_ag_single_dinov2
from .ag_single_dinov2_prev import build as build_ag_single_dinov2_prev # load t-1 previous frame
from .ag_single_dinov2_prev2 import build as build_ag_single_dinov2_prev2

from .ag_multi_dinov2 import build as build_ag_multi_dinov2
from .ag_multi_dinov2_prev import build as build_ag_multi_dinov2_prev # load t-1 previous frame

def build_dataset(image_set, args):
    if args.dataset_file == 'ag_single':
        return build_ag_single(image_set, args)
    if args.dataset_file == 'ag_multi':
        return build_ag_multi(image_set, args)
        
    if args.dataset_file == 'ag_single_dinov2': # dino backbone
        # return build_ag_single_dinov2(image_set, args)
        # return build_ag_single_dinov2_prev(image_set, args)
        return build_ag_single_dinov2_prev2(image_set, args)
    if args.dataset_file == 'ag_multi_dinov2':
        # return build_ag_multi_dinov2(image_set, args)
        return build_ag_multi_dinov2_prev(image_set, args)
        
    raise ValueError(f'dataset {args.dataset_file} not supported')


# def build_dataset(image_set, args):
#     if args.dataset_file == 'ag_single':
#         return build_ag_single(image_set, args)
#     if args.dataset_file == 'ag_multi':
#         return build_ag_multi(image_set, args)

#     if args.dataset_file == 'ag_single_dinov2': # dinov2 backbone
#         # return build_ag_single_dinov2(image_set, args)
#         # return build_ag_single_dinov2_prev(image_set, args)
#         return build_ag_single_dinov2_prev2(image_set, args)
#     if args.dataset_file == 'ag_multi_dinov2':
#         # import pdb;pdb.set_trace()
#         # return build_ag_multi_dinov2(image_set, args)
#         return build_ag_multi_dinov2_prev(image_set, args)
    
#     raise ValueError(f'dataset {args.dataset_file} not supported')
