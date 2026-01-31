from .dsgg_single_sgdet import build as build_single_sgdet
from .dsgg_single_sgdet_Diff2 import build as build_single_sgdet_Diff2

from .dsgg_single_xcls import build as build_single_xcls

from .dsgg_multi_sgdet import build as build_multi_sgdet
from .dsgg_multi_sgdet_Diff2 import build as build_multi_sgdet_Diff2
from .dsgg_multi_sgdet_batches import build as build_multi_sgdet_batches

from .dsgg_multi_xcls import build as build_multi_xcls
from .dsgg_multi_2 import build as build_multi_2

def build_model(args):
    if args.dataset_file == 'ag_single':
        if args.dsgg_task == 'sgdet': # spatial
            return build_single_sgdet_Diff2(args)
        else:
            return build_single_xcls(args)
        
    elif args.dataset_file in ['ag_single_dinov2', 'ag_single_dinov3']:
        if args.dsgg_task == 'sgdet': # spatial
            # return build_single_sgdet(args)
            return build_single_sgdet_Diff2(args)
    
    elif args.dataset_file == 'ag_multi':
        if args.dsgg_task == 'sgdet':
            return build_multi_sgdet(args)
        else:
            return build_multi_xcls(args)
        
    elif args.dataset_file in ['ag_multi_dinov2', 'ag_multi_dinov3']:
        if args.dsgg_task == 'sgdet':
            return build_multi_sgdet_Diff2(args)
            # return build_multi_sgdet_batches(args)
        else:
            return build_multi_xcls(args)
    else:
        raise NotImplementedError

# def build_model(args):
#     if args.dataset_file == 'ag_single':
#         if args.dsgg_task == 'sgdet':
#             return build_single_sgdet(args)
#         else:
#             # if args.one_dec and args.dsgg_task == 'predcls':
#             #     return buidl_single_predcls_one_dec(args)
#             # else:
#             return build_single_xcls(args)
#     elif args.dataset_file == 'ag_multi':
#         if args.dsgg_task == 'sgdet':
#             if args.method2: # temp trying
#                 return build_multi_2(args)
#             return build_multi_sgdet(args)
#         else:
#             return build_multi_xcls(args)
#     else:
#         raise NotImplementedError
    
    