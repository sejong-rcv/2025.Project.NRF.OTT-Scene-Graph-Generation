import math
import os
import sys
from typing import Iterable
import numpy as np
import copy
import itertools

import torch
from torch.cuda.amp import autocast, GradScaler
import pickle

import util.misc as utils
# from models.evaluate_recall import BasicSceneGraphEvaluator
from models.evaluate_recall_rel_upper import BasicSceneGraphEvaluator

import cv2
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from thop import profile
from tqdm import tqdm
import time
from pathlib import Path

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0, 
                    args=None, writer = None, lr_scheduler=None):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    if hasattr(criterion, 'loss_labels'):
        metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    elif 'obj_labels' in criterion.losses:
        metric_logger.add_meter('obj_class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    
    print_freq = 10 
    
    maxIter = len(data_loader)
    cur_iter = epoch * maxIter
    
    optimizer.zero_grad(set_to_none=True)
    update_step = 0
    
    # scaler = GradScaler() # bfloat 사용하므로 amp에 gradscaler 사용X
    
    for i, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        samples = samples.to(device)
        # if type(targets[0]) == dict: # targets : dict 담은 list(batch size의 길이)
        if isinstance(targets[0], dict): # 단일 target: List[Dict]
            #  targets = [{k: v.to(device) for k, v in t.items() if k != 'filename'} for t in targets] # load targets to device
            targets = [_move_target_dict_to_device(t, device) for t in targets]
        elif isinstance(targets[0], list) and len(targets[0]) == 2 and isinstance(targets[0][0], dict): # target, prev_target
            # targets: List[[cur_dict, prev_dict], ...]
            targets = [
                [_move_target_dict_to_device(cur_t, device), _move_target_dict_to_device(prev_t, device),]
                for (cur_t, prev_t) in targets] # move to device
        else:
            raise TypeError(f"Unexpected targets type: {type(targets[0])}")

        with autocast(dtype=torch.bfloat16):
            in_targets = targets if (args.dsgg_task != 'sgdet' or args.use_matched_query) else None
            if 'cur_idx' in targets[0].keys(): # 단일 target
            # if 'cur_idx' in targets[0][0].keys():  # multi,sgcls [cur, prev] target. _prev dataset
                cur_idx = targets[0]['cur_idx'].item()
                outputs = model(samples, targets=in_targets, cur_idx=cur_idx)
            else:
                outputs = model(samples, targets=in_targets)
            if len(targets) > 1 and args.dsgg_task == 'sgdet' and args.dataset_file == 'ag_multi':
                import pdb;pdb.set_trace()
                targets = [targets[0]]
            
            loss_dict = criterion(outputs, targets)
            weight_dict = criterion.weight_dict
            losses = sum((loss_dict[k] * weight_dict[k]).float()
                         for k in loss_dict.keys() if k in weight_dict)
        
        # # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)
        
        losses = losses / max(1, args.accum_iter)
        
        losses.backward() 
        
        if (i + 1) % args.accum_iter == 0:
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)  
        
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            
            if lr_scheduler is not None:
                lr_scheduler.step()
            update_step += 1
                
        # metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        if hasattr(criterion, 'loss_labels'):
            metric_logger.update(class_error=loss_dict_reduced['class_error'])
        elif 'obj_class_error' in loss_dict_reduced.keys():
            metric_logger.update(obj_class_error=loss_dict_reduced['obj_class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(
            loss=loss_value,                    # Total Loss (Scaled)
            attn=loss_dict_reduced.get('loss_attn_ce', 0),      # Attention Relation Loss
            spatial=loss_dict_reduced.get('loss_spatial_ce', 0), # Spatial Loss
            contact=loss_dict_reduced.get('loss_contacting_ce', 0)  # Contacting Loss
        )
    
        if writer is not None:
            writer.add_scalar('loss/total_loss', loss_value, cur_iter)
            for k, v in loss_dict_reduced.items():
                writer.add_scalar('loss/{}'.format(k), v, cur_iter)
        cur_iter += 1
        
    if (i + 1) % args.accum_iter != 0: 
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if lr_scheduler is not None:
            lr_scheduler.step()

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate_dsgg(dataset_file, model, postprocessors, data_loader, device, args):
    model.eval()
    print('  checkpoint :', args.pretrained)
    # ex) exps/exp1/eval_log_sgdet_rank0.txt
    ckpt_dir = os.path.dirname(args.pretrained)
    rank = utils.get_rank()
    ckpt_tag = Path(args.pretrained).stem          # checkpoint_6
    run_id = f"pid{os.getpid()}_{time.time_ns()}"  # 프로세스/실행 고유

    def get_log_path(mode):
        return os.path.join(
            ckpt_dir,
            f".tmp_eval_{ckpt_tag}_{args.dsgg_task}_{mode}_rank{rank}_{run_id}.log"
            )
    
    # evaluator1 = BasicSceneGraphEvaluator(mode=args.dsgg_task, iou_threshold=0.5, constraint='with', nms_filter=args.use_nms_filter)
    # evaluator2 = BasicSceneGraphEvaluator(mode=args.dsgg_task, iou_threshold=0.5, constraint='no', nms_filter=args.use_nms_filter)
    
    evaluator1 = BasicSceneGraphEvaluator(mode=args.dsgg_task, iou_threshold=0.5, constraint='with', 
                                          nms_filter=args.use_nms_filter, upper_bound=False,
                                          save_file=get_log_path('with')) 
    evaluator2 = BasicSceneGraphEvaluator(mode=args.dsgg_task, iou_threshold=0.5, constraint='no', 
                                          nms_filter=args.use_nms_filter, upper_bound=False,
                                          save_file=get_log_path('no'))  
    # evaluator_upper = BasicSceneGraphEvaluator(mode='sgdet',   iou_threshold=0.5, constraint='with',
    #                                            nms_filter=args.use_nms_filter, upper_bound=True,
    #                                            save_file=get_log_path('upper')) # 고유 경로 전달

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    save_prediction = False
    preds_dict = {}
    to_test = 100

    for samples, targets in metric_logger.log_every(data_loader, 100, header):
        samples = samples.to(device)
        
        if isinstance(targets[0], (list, tuple)) and len(targets[0]) == 2:
            cur_targets = [t[0] for t in targets]
            prev_targets = [t[1] for t in targets]
            targets = cur_targets
        else:
            prev_targets = [None for _ in targets]
        
        if type(targets[0]) == list:
            targets = [{k: v.to(device) if type(v) == torch.Tensor else v for k, v in t.items() if k != 'filename'} for target in targets for t in target]           
        in_targets = targets if (args.dsgg_task != 'sgdet' or args.use_matched_query) else None

        if 'cur_idx' in targets[0].keys():
            cur_idx = targets[0]['cur_idx'].item()
            outputs = model(samples, targets=in_targets, cur_idx=cur_idx)
        else:
            outputs = model(samples, targets=in_targets)
        if len(targets) > 1 and args.dsgg_task == 'sgdet' and args.dataset_file == 'ag_multi':
            targets = [targets[0]]

        
        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)

        if args.dsgg_task == 'sgdet':
            results = postprocessors['dsgg'](outputs, orig_target_sizes)
        else:
            cur_idx = 0
            if args.dataset_file == 'ag_multi':
                cur_idx = targets[0]['cur_idx']
                targets = [targets[cur_idx]]
            if args.seq_sort:
                results = postprocessors['dsgg'](outputs, targets, cur_idx)
            else:
                results = postprocessors['dsgg'](outputs, targets)
        if save_prediction:
            preds_dict[targets[0]['img_path']] = results
       
        evaluator1.evaluate_scene_graph(targets, results)
        evaluator2.evaluate_scene_graph(targets, results)
        evaluator_upper.evaluate_scene_graph(targets, results)
        
        # to_test -= 1
        # if to_test == 0:
        #     break
    metric_logger.synchronize_between_processes()

    # if save_prediction:
    #     with open('sgdet_single_preds_dict.pkl', 'wb') as f:
    #         pickle.dump(preds_dict, f)
    print('  checkpoint :', args.pretrained)
    stats = {}
    print('-------------------------with constraint-------------------------------')
    stats['with'] = evaluator1.print_stats()
    print('-------------------------no constraint-------------------------------')
    stats['no'] = evaluator2.print_stats()
    print('-------------------------relation upper bound-------------------------------')
    stats['upper'] = evaluator_upper.print_stats()
    
    
    ### save inference results
    if utils.is_main_process():
        output_dir = os.path.dirname(args.pretrained)
        ckpt_tag = Path(args.pretrained).stem
        run_id = f"pid{os.getpid()}_{time.time_ns()}"
        
        save_path = os.path.join(output_dir, f"inference_result_{ckpt_tag}_{run_id}.txt")

        with open(save_path, "w") as f:
            f.write(f"=== OED Video SGG Evaluation Results ===\n")
            f.write(f"Checkpoint: {args.pretrained}\n")
            f.write(f"Task: {args.dsgg_task} | Backbone: {args.backbone}\n")
            f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            K_list = [10, 20, 50, 100]

            for mode in ['with', 'no', 'upper']:
                f.write(f"[{mode.upper()} CONSTRAINT]\n")
                current_mode_stats = stats.get(mode, {})
                
                # Recall@K
                for k in K_list:
                    r_val = current_mode_stats.get(f'R@{k}', 0.0)
                    f.write(f"  R@{k:<3}: {r_val:.4f}\n")
                
                # mean Recall@K 
                if mode != 'upper':
                    for k in K_list:
                        mr_val = current_mode_stats.get(f'mR@{k}', 0.0)
                        f.write(f"  mR@{k:<2}: {mr_val:.4f}\n")
                f.write("\n")
            
            f.write("="*40 + "\n\n")
            f.write(f"args : {args}\n")
        print(f"Full stats saved to: {save_path}")
        
    for mode in ['with', 'no', 'upper']: 
        tmp_p = get_log_path(mode)
        if os.path.exists(tmp_p): os.remove(tmp_p)

    return stats

def plot_attn_weight(attention_mask, img_path, post_name):
    """
    img_path:   image file path to load
    save_path:  image file path to save
    attention_mask:  2-D attention map with np.array type, e.g, (h, w) or (w, h)
    ratio:  scaling factor to scale the output h and w
    cmap:  attention style, default: "jet"
    quality:  saved image quality
    """
    img_name = img_path.split('/')[1].split('.')[0] + '_' + img_path.split('/')[2].split('.')[0]
    img_path = './data/action-genome/' + img_path
    print("load image from: ", img_path)
    # img = Image.open(img_path, mode='r')
    # img_w, img_h = img.size[0], img.size[1]
    img = cv2.imread(img_path)
    img_h, img_w = img.shape[:2]


    plt.subplots(nrows=1, ncols=1, figsize=(0.02 * img_w, 0.02 * img_h))



    plt.imshow(img, alpha=1) 
    plt.axis('off') 

    mask = cv2.resize(attention_mask, (img_w, img_h))    
    normed_mask = mask / mask.max() #
    normed_mask = (normed_mask * 255).astype('uint8')  
    
    plt.imshow(normed_mask, alpha=0.5, interpolation='nearest')

    # build save path
    save_folder = os.path.join('./visual_attn_obj_score/', img_name)
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    save_path = os.path.join(save_folder, post_name+'.jpg')
    ori_path = os.path.join(save_folder, 'orig.jpg')
    
    print("save image to: " + save_path)
    plt.axis('off')
    plt.subplots_adjust(top=1, bottom=0, right=1,  left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    quality = 200
    plt.savefig(save_path, dpi=quality)

    if not os.path.exists(ori_path):
        # save original image file
        cv2.imwrite(ori_path, img)


def evaluate_speed(dataset_file, model, postprocessors, data_loader, device, args):
    model.eval()
    total_time = 0
    warmup = 5
    data_loader2 = copy.deepcopy(data_loader)
    all_flops = 0
    model = model.half()
    
    print('--------- starting warm-up ---------')
    for i, (w_samples, targets) in enumerate(tqdm(data_loader2)):
        w_samples.tensors = w_samples.tensors.half()
        w_samples = w_samples.to(device)
        
        #  if type(targets[0]) == list:
        if isinstance(targets[0], list): # 수정
            targets = [{k: v.to(device) if type(v) == torch.Tensor else v for k, v in t.items() if k != 'filename'} for target in targets for t in target]
        # flops, params = profile(model, (w_samples,))
        # print('flops: ', flops, 'params: ', params)
        # print('flops: %.2f G, params: %.2f M' % (flops / 1000000000.0, params / 1000000.0))
        
        # with torch.no_grad():
        outputs = model(w_samples, targets=None)
        warmup -= 1
        if warmup < 0:
            break
    print('--------- ending warm-up ---------')
    repetitions = 10
    total_frame = 0
    total_time = 0
    bs = 64
    print('--------- starting running ---------')
    rep = 0
    for i, (samples, targets) in enumerate(tqdm(data_loader)):
        samples.tensors = samples.tensors.half()
        samples = samples.to(device)
        if type(targets[0]) == list:
            targets = [{k: v.to(device) if type(v) == torch.Tensor else v for k, v in t.items() if k != 'filename'} for target in targets for t in target]
        in_targets = targets if (args.dsgg_task != 'sgdet' or args.use_matched_query) else None
        
        start = time.time()
        with torch.no_grad():
            outputs = model(samples, targets=in_targets)
        end = time.time()
        total_time += (end - start)
        total_frame += bs
        
        rep += 1
        if rep >= repetitions:
            break
    print(total_time * 1000 / total_frame)

def _move_target_dict_to_device(t: dict, device):
    return {
        k: (v.to(device) if torch.is_tensor(v) else v)
        for k, v in t.items()
    }