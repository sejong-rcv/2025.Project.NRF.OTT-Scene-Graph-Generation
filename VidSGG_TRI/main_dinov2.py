import argparse
import time
import datetime
import random
from pathlib import Path
import json
import os
import math

import numpy as np
import torch
import torch.distributed as dist
import torch.cuda.amp as amp
from torch.utils.data import DataLoader, DistributedSampler

import datasets
from datasets import build_dataset

from engine import train_one_epoch, evaluate_dsgg, evaluate_speed

from models import build_model
import os
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter

def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    
    # Learning Rate Scheduling & Optimzier
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=0, type=float)
    parser.add_argument('--lr_drop', default=[60], type=int, nargs='+',
                        help='epochs at which to drop learning rate (e.g., --lr_drop 25 30)')
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=90, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--is_train', default='test', type=str)
    parser.add_argument('--accum_iter', default=1, type=int,
                    help='gradient accumulation steps (effective batch multiplier)')
    parser.add_argument('--lr_base_batch', default=64, type=int,
                        help='reference effective batch size for lr scaling')
    parser.add_argument('--scale_lr', action='store_true', default=True,
                        help='enable linear lr scaling by effective batch')
    # scheduler options
    parser.add_argument('--sched', default='cosine', type=str, choices=['cosine', 'step'],
                        help='lr scheduler type')
    parser.add_argument('--warmup_ratio', default=0.05, type=float,
                        help='warmup steps ratio over total optimizer steps')
    parser.add_argument('--min_lr_ratio', default=0.01, type=float,
                        help='min lr ratio for cosine schedule (final_lr = base_lr * ratio)')
    parser.add_argument('--step_gamma', default=0.1, type=float,
                        help='gamma for StepLR if sched=step')  
    
    # Model parameters
    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers_hopd', default=3, type=int,
                        help="Number of hopd decoding layers in the transformer")
    parser.add_argument('--dec_layers_interaction', default=3, type=int,
                        help="Number of interaction decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=100, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')
    parser.add_argument('--masks', action='store_true', help="Train segmentation head if the flag is provided")
    
    # HOI
    parser.add_argument('--num_obj_classes', type=int, default=36, # TODO
                        help="Number of object classes")
    parser.add_argument('--num_verb_classes', type=int, default=117,
                        help="Number of verb classes")
    parser.add_argument('--pretrained', type=str, default='',
                        help='Pretrained model path')
    parser.add_argument('--subject_category_id', default=1, type=int)
    parser.add_argument('--verb_loss_type', type=str, default='focal',
                        help='Loss type for the verb classification')

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    parser.add_argument('--use_matching', action='store_true',
                        help="Use obj/sub matching 2class loss in first decoder, default not use")

    # * Matcher
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=2.5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=1, type=float,
                        help="giou box coefficient in the matching cost")
    parser.add_argument('--set_cost_obj_class', default=1, type=float,
                        help="Object class coefficient in the matching cost")
    parser.add_argument('--set_cost_verb_class', default=1, type=float,
                        help="Verb class coefficient in the matching cost")
    parser.add_argument('--set_cost_matching', default=1, type=float,
                        help="Sub and obj box matching coefficient in the matching cost")
    parser.add_argument('--set_cost_rel_class', default=1, type=float)

    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--bbox_loss_coef', default=2.5, type=float)
    parser.add_argument('--giou_loss_coef', default=1, type=float)
    parser.add_argument('--obj_loss_coef', default=1, type=float)
    parser.add_argument('--verb_loss_coef', default=2, type=float)
    parser.add_argument('--rel_loss_coef', default=2, type=float)
    parser.add_argument('--alpha', default=0.5, type=float, help='focal loss alpha')
    parser.add_argument('--matching_loss_coef', default=1, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")

    # dataset parameters
    parser.add_argument('--dataset_file', default='ag_single')
    parser.add_argument('--ag_path', type=str)
    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=4, type=int)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    # decoupling training parameters
    parser.add_argument('--freeze_mode', default=0, type=int) # freeze parameters other than those of temporal module and head

    # hoi eval parameters
    parser.add_argument('--use_nms_filter', action='store_true', help='Use pair nms filter, default not use')
    parser.add_argument('--thres_nms', default=0.7, type=float)
    parser.add_argument('--nms_alpha', default=1.0, type=float)
    parser.add_argument('--nms_beta', default=0.5, type=float)
    parser.add_argument('--json_file', default='results.json', type=str)

    parser.add_argument('--cache_mode', action='store_true', default=False)

    # dsgg parameters
    parser.add_argument('--num_attn_classes', default=3, type=int)
    parser.add_argument('--num_spatial_classes', default=6, type=int)
    parser.add_argument('--num_contacting_classes', default=17, type=int)
    parser.add_argument('--dsgg_task', default='sgdet', type=str)
    parser.add_argument('--interval1', default=4, type=int)
    parser.add_argument('--interval2', default=4, type=int)
    parser.add_argument('--num_ref_frames', default=4, type=int) # 원래 default 2
    parser.add_argument('--dec_layers_temporal', default=1, type=int)
    parser.add_argument('--seq_sort', action='store_true', default=False)
    parser.add_argument('--fuse_semantic_pos', action='store_true', default=False)

    parser.add_argument('--temporal_feature_encoder', action='store_true', default=False)
    parser.add_argument('--instance_temporal_interaction', action='store_true', default=False)
    parser.add_argument('--relation_temporal_interaction', action='store_true', default=False)
    parser.add_argument('--query_temporal_interaction', action='store_true', default=False) # concat pair query and relation query together to perform temporal query interaction
    parser.add_argument('--temporal_embed_head', action='store_true', default=False) # use temporal head and keep spatial head frozen
    parser.add_argument('--temporal_decoder_init', action='store_true', default=False) # use spatial decoder weight to initialize the weight of temporal decoder 
    parser.add_argument('--one_dec', action='store_true', default=False) # only use one decoder, view predcls as a predicate classification problem
    parser.add_argument('--use_roi', action='store_true', default=False) # use RoI feature to initialize query feature
    parser.add_argument('--one_temp', action='store_true', default=False) # only use one temporal query interaction, not in a coarse-to-fine way
    parser.add_argument('--no_update_pair', action='store_true', default=False) # concat query to temporal interaction, but only update relation query
    parser.add_argument('--semantic_head', action='store_true', default=False)
    parser.add_argument('--use_matched_query', action='store_true', default=False)
    parser.add_argument('--aux_learnable_query', action='store_true', default=False)

    parser.add_argument('--obj_reweight', action='store_true', default=False)
    parser.add_argument('--rel_reweight', action='store_true', default=False)
    parser.add_argument('--use_static_weights', action='store_true', default=False)
    parser.add_argument('--queue_size', default=4704*1.0, type=float,
                        help='Maxsize of queue for obj and verb reweighting, default 1 epoch')
    parser.add_argument('--p_obj', default=0.7, type=float,
                        help='Reweighting parameter for obj')
    parser.add_argument('--p_rel', default=0.7, type=float,
                        help='Reweighting parameter for verb')

    return parser


def load_state_dict_safely(model, checkpoint_state_dict, verbose=True):
    model_state = model.state_dict()
    filtered = {}
    skipped = []
    # Key matching, Shape checking. 
    
    for k, v in checkpoint_state_dict.items():
        if k in model_state and model_state[k].shape == v.shape:
            filtered[k] = v
        else:
            skipped.append(k)
    msg = model.load_state_dict(filtered, strict=False)
    if verbose:
        print("[StateDict] loaded keys:", len(filtered), "; skipped keys:", len(skipped))
        if skipped:
            print("  skipped (shape mismatch or missing in model):")
            for kk in skipped[:20]:  
                print("   -", kk, "ckpt:", checkpoint_state_dict[kk].shape,
                      "| model:", model_state.get(kk, torch.empty(0)).shape if kk in model_state else "N/A")
            if len(skipped) > 20:
                print("   ... ({} more)".format(len(skipped)-20))
    return msg

def main(args):
    if args.dataset_file in ['ag_single', 'ag_single_dinov2', 'ag_single_dinov3']:
        import util.misc as utils
    elif args.dataset_file in ['ag_multi', 'ag_multi_dinov2', 'ag_multi_dinov3']:
        import util.misc_multi as utils
    else:
        raise AssertionError('No support for this dataset')


    # read informabltions from DDP environment variable
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ and 'LOCAL_RANK' in os.environ and int(os.environ['WORLD_SIZE'])>1:
        args.rank = int(os.environ['RANK'])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
        args.distributed = True
    else:     
        args.rank = 0
        args.world_size = 1
        args.gpu = 0
        args.distributed = False
    
    if args.distributed: 
        torch.cuda.set_device(args.gpu)

    if args.distributed: # process group init.
        local_world = os.environ.get('LOCAL_WORLD_SIZE')
        if local_world is None:
            local_world = str(torch.cuda.device_count())  
        os.environ.setdefault('LOCAL_SIZE', local_world)
        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=args.world_size,
            rank=args.rank
        )
        utils.setup_for_distributed(args.rank == 0)
        _ = torch.tensor(0, device=f'cuda:{args.gpu}')
        dist.barrier(device_ids=[args.gpu])
        print(f'| distributed init (rank {args.rank} / world {args.world_size})', flush=True)
    
    print(args)
    assert args.dsgg_task in ['sgdet', 'sgcls', 'predcls']
    device = torch.device(f'cuda:{args.gpu}' if args.distributed else args.device)

    # TODO random
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
        
    # Build DataLoader    
    dataset_train = build_dataset(image_set='train', args=args)
    dataset_val = build_dataset(image_set='val', args=args)

    if args.distributed:
        sampler_train = DistributedSampler(dataset_train)
        sampler_val = DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)

    data_loader_train = DataLoader(
        dataset_train,
        batch_sampler=batch_sampler_train,
        collate_fn=utils.collate_fn, 
        num_workers=args.num_workers,
        persistent_workers=True if args.num_workers > 0 else False,
        pin_memory=True
        )
    
    data_loader_val = DataLoader(
        dataset_val, 
        args.batch_size, 
        sampler=sampler_val,
        drop_last=False, 
        collate_fn=utils.collate_fn, 
        num_workers=args.num_workers, 
        persistent_workers=True if args.num_workers > 0 else False,
        pin_memory=True
        )    
    
    iters_per_epoch = len(data_loader_train)

    updates_per_epoch = math.ceil(iters_per_epoch / max(1, args.accum_iter))
    total_updates = args.epochs * updates_per_epoch
    warmup_steps = int(args.warmup_ratio * total_updates)
    
    # total_updates = (args.epochs * iters_per_epoch) // max(1, args.accum_iter)
    # warmup_steps = int(args.warmup_ratio * total_updates)

    # 실제 batch size 계산(per-gpu batch x world_size)
    effective_batch = args.batch_size * args.world_size * args.accum_iter
    
    if args.scale_lr:
        lr_scale = effective_batch / float(args.lr_base_batch)
    else:
        lr_scale = 1.0

    scaled_lr = args.lr * lr_scale
    scaled_lr_backbone = args.lr_backbone * lr_scale

    # Build Model
    # if dsgg_task is 'sgdet', 'models/dsgg_single_sgdet' or 'models/dsgg_multi_sgdet'
    model, criterion, postprocessors = build_model(args)
    model.to(device)

    model_without_ddp = model

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, 
            device_ids=[args.gpu], 
            output_device=args.gpu,
            find_unused_parameters=False,
            static_graph=True
            )
        model_without_ddp = model.module
            
    print(f"[rank {args.rank}] LOCAL_RANK={args.gpu}, "
      f"current_device={torch.cuda.current_device()}, "
      f"device_name={torch.cuda.get_device_name(torch.cuda.current_device())}, "
      f"visible={os.environ.get('CUDA_VISIBLE_DEVICES')}")
                
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if args.freeze_mode == 1: # train/eval temporal_predcls/sgdet
        # freeze spatial baseline but embed head
        print(" Freeze_Mode 1: Training Temporal & Motion Modules")

        # print('================ Tuning the parameters below =================')
        for name, p in model_without_ddp.named_parameters():
            if 'temporal' in name or 'semantic_pos' in name or \
                    (args.aux_learnable_query and 'query_embed' in name):   
                p.requires_grad = True
                if utils.get_rank() == 0:
                    print('train :', name)
                
            elif not args.query_temporal_interaction and ((not args.instance_temporal_interaction and ('bbox_embed' in name or 'obj_class_embed' in name)) or \
              (not args.relation_temporal_interaction and 'class_embed' in name and 'obj_class_embed' not in name)):
                print('Train : ', name)
                p.requires_grad = True
            else:
                if utils.get_rank() == 0:
                    print("Freeze :", name)
                p.requires_grad = False # Freeze
            
    param_dicts = [
        {"params": [p for n, p in model_without_ddp.named_parameters()
                    if "backbone" not in n and p.requires_grad],
        "lr": scaled_lr
        },
        {"params": [p for n, p in model_without_ddp.named_parameters() 
                    if "backbone" in n and p.requires_grad],
        "lr": scaled_lr_backbone,
        },
    ]
    
    # optimizer = torch.optim.AdamW(params=param_dicts, lr=args.lr, weight_decay=args.weight_decay)
    optimizer = torch.optim.AdamW(param_dicts, weight_decay=args.weight_decay)

    # scheduler (iteration-based)
    if args.sched == 'cosine':
        def lr_lambda(step):
            u = step + 1  # 1-based update index
            # warmup
            if warmup_steps > 0 and u <= warmup_steps:
                return float(u) / float(max(1, warmup_steps))
            
            t = u - warmup_steps - 1            
            # cosine decay after warmup
            denom = float(max(1, total_updates - warmup_steps - 1))
            progress = min(max(t / denom, 0.0), 1.0)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return args.min_lr_ratio + (1.0 - args.min_lr_ratio) * cosine
        
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        
    elif args.sched == 'step':
        # step_size_updates = max(1, args.lr_drop * updates_per_epoch)
        milestones = args.lr_drop if isinstance(args.lr_drop, list) else [args.lr_drop]
        
        def lr_lambda(step):
            u = step + 1 # 1-based index

            if warmup_steps > 0 and u <= warmup_steps:
                return float(u) / float(max(1, warmup_steps))
            curr_epoch = step // updates_per_epoch

            exp = sum(curr_epoch >= m for m in milestones)
            
            return args.step_gamma ** exp
        
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    if lr_scheduler is not None and (not args.resume) and (not args.eval):
        lr_scheduler.step()  # 첫 optimizer.step부터 warmup lr 적용

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        load_state_dict_safely(model_without_ddp, checkpoint['model'], verbose=True)
        if not args.eval and 'epoch' in checkpoint:
            args.start_epoch = checkpoint['epoch'] + 1
            print(f"[Resume] Weights loaded. Starting from epoch {args.start_epoch}")


    elif args.pretrained:
        checkpoint = torch.load(args.pretrained, map_location='cpu')
        if args.eval:
            # model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
            load_state_dict_safely(model_without_ddp, checkpoint['model'], verbose=True)
        else: # train
            assert not args.instance_temporal_interaction
            temp_dict = {}
            remove_list = []
            for name in checkpoint['model'].keys():
                if args.dataset_file in ['ag_single', 'ag_single_dinov2', 'ag_single_dinov3']: # spatial
                    if args.semantic_head and 'class' in name:
                        remove_list.append(name)                      
                elif args.dataset_file in ['ag_multi', 'ag_multi_dinov2', 'ag_multi_dinov3']: # temporal
                    replaced_name = None
                    if args.temporal_decoder_init: # load temporal decoder weights
                        if 'transformer.decoder.layers.2' in name:
                            replaced_name = name.replace('transformer.decoder.layers.2', 'transformer.ins_temporal_interaction_decoder.layers.0')
                        if 'transformer.interaction_decoder.layers.2' in name:
                            replaced_name = name.replace('transformer.interaction_decoder.layers.2', 'transformer.rel_temporal_interaction_decoder.layers.0')
                    if args.temporal_embed_head and args.relation_temporal_interaction:
                        if 'class_embed' in name and not 'obj_class_embed' in name: # relation classification
                            replaced_name = name.replace(name, 'temporal_' + name)
                    if args.query_temporal_interaction: # sgdet multi에서 activate
                        if 'class_embed' in name or 'bbox_embed' in name:
                            replaced_name = name.replace(name, 'temporal_' + name) # ex) obj_class_embed.weight -> temporal_obj_class_embed.weight
                    if replaced_name is not None:
                        # print('name: ', name, '; replaced_name: ', replaced_name)
                        # assert replaced_name in model_without_ddp.state_dict().keys()
                        temp_dict[replaced_name] = checkpoint['model'][name] 
                        # if replaced_name not in model_without_ddp.state_dict().keys():       
            
            checkpoint['model'].update(temp_dict)
            for name in remove_list:
                del checkpoint['model'][name]
            load_state_dict_safely(model_without_ddp, checkpoint['model'], verbose=True)
            # model_without_ddp.load_state_dict(checkpoint['model'], strict=False) # load pretrained weight
    if args.eval:
        test_stats = evaluate_dsgg(args.dataset_file, model, postprocessors, data_loader_val, device, args)
        # evaluate_speed(args.dataset_file, model, postprocessors, data_loader_val, device, args)
        return
    
    output_dir = Path(args.output_dir)
    
    if (not args.distributed) or (args.rank == 0):
        print(f"[LR] iters/epoch={iters_per_epoch}, updates/epoch={updates_per_epoch}, total_updates={total_updates}")
        print(f"[LR] warmup_steps={warmup_steps}, effective_batch={effective_batch}, lr_scale={lr_scale}")
        print(f"[LR] base_lr={args.lr}, scaled_lr={scaled_lr}, scaled_lr_backbone={scaled_lr_backbone}")

    print("Start training")
    start_time = time.time()
    best_performance = 0
    writer = None
    if 'pdb' not in args.output_dir:
        writer = SummaryWriter(args.output_dir + 'curve') if utils.is_main_process() else None
        # writer = SummaryWriter(args.output_dir + 'curve')
        
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            sampler_train.set_epoch(epoch)
        train_stats = train_one_epoch( # train 
            model, criterion, data_loader_train, optimizer, device, epoch,
            args.clip_max_norm, args, writer, lr_scheduler=lr_scheduler)
        
        if epoch == args.epochs - 1:
            checkpoint_path = os.path.join(output_dir, 'checkpoint_last.pth')
            utils.save_on_master({
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'args': args,
            }, checkpoint_path)

        # new added save script
        # 'ag_single' : spatial / 'ag_multi' : temporal
        # if get_rank is 0, torch.save(dict, checkpoint_path)
        if (args.dataset_file in ['ag_single', 'ag_single_dinov2', 'ag_single_dinov3', 
                                  'ag_multi', 'ag_multi_dinov2', 'ag_multi_dinov3'] and epoch % 1 == 0):
            checkpoint_path = os.path.join(output_dir, 'checkpoint_{}.pth'.format(epoch))
            utils.save_on_master({
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'args': args,
            }, checkpoint_path)

        ## evaluation one epoch every some epochs and save the best model accoding to the evaluation metic
        # if args.freeze_mode == 0 and epoch < args.lr_drop and epoch % 5 != 0:  ## eval every 5 epoch before lr_drop
        #     continue
        # elif args.freeze_mode == 0 and epoch >= args.lr_drop and epoch % 2 == 0:  ## eval every 2 epoch after lr_drop
        #     continue
        # test_stats = evaluate_dsgg(args.dataset_file, model, postprocessors, data_loader_val, device, args)
        # test_with_stats = test_stats['with']
        # test_semi_stats = test_stats['semi']
        # test_no_stats = test_stats['no']

        # if performance > best_performance:
        #     checkpoint_path = os.path.join(output_dir, 'checkpoint_best.pth')
        #     utils.save_on_master({
        #         'model': model_without_ddp.state_dict(),
        #         'optimizer': optimizer.state_dict(),
        #         'lr_scheduler': lr_scheduler.state_dict(),
        #         'epoch': epoch,
        #         'args': args,
        #     }, checkpoint_path)
        
        #     best_performance = performance

        log_stats = {'epoch': epoch,
                    **{f'train_{k}': v for k, v in train_stats.items()},
                     # **{f'test_with_{k}': v for k, v in test_with_stats.items()},
                     # **{f'test_semi_{k}': v for k, v in test_semi_stats.items()},
                     # **{f'test_no_{k}': v for k, v in test_no_stats.items()},
                    "effective_batch": effective_batch,
                    "batch_size_per_gpu": args.batch_size,
                    "world_size": args.world_size,
                    "scaled_lr_init": scaled_lr,
                    'n_parameters': n_parameters}

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

    writer.close()
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

def name_match_baseline(name):
    baseline_modules_without_head = ['transformer.encoder.', 'transformer.decoder.', 'transformer.interaction_decoder.', 
                       'input_proj.', 'query_embed.', 'backbone.']
    for module in baseline_modules_without_head:
        if module in name:
            return True
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser('OED training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
