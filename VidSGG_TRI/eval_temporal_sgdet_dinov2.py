import os
import sys
import random

## check backbone, dataset
# check models/__init__.py model import

def run_eval(device='0', ckpt_path='exps/temporal_sgdet/checkpoint_5.pth'):
    port = random.randint(10000,59999)
    num_devices = len(device.split(','))
    os.environ['CUDA_VISIBLE_DEVICES'] = device
    os.system(f"python -m torch.distributed.launch \
                    --master_addr 127.0.0.1 \
                    --master_port {port} \
                    --nproc_per_node={num_devices} \
                    --use_env \
                    main_dinov2.py \
                    --dataset_file ag_multi_dinov2 \
                    --ag_path data/action-genome \
                    --dec_layers_hopd 6 \
                    --dec_layers_interaction 6 \
                    --eval \
                    --batch_size 16 \
                    --query_temporal_interaction \
                    --use_nms_filter \
                    --pretrained {ckpt_path} \
                    --dsgg_task sgdet \
                    --num_workers 2 \
                    --num_ref_frames 2 \
                    --backbone vit_base_patch14_dinov2 \
                    --num_queries 100 \
                    ")

if __name__ == "__main__":
    device = '1'
    run_eval(device=device, ckpt_path='exps/temporal_sgdet_vitB_batches_reproduce2/checkpoint_9.pth')