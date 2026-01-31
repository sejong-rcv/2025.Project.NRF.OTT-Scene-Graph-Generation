import os
import sys
import random

## check backbone, num_queries, dataset 

def run_eval(device='0', ckpt_path=''):
    port = random.randint(10000,59999)
    num_devices = len(device.split(','))
    os.environ['CUDA_VISIBLE_DEVICES'] = device
    os.system(f"python -m torch.distributed.launch \
                    --master_addr 127.0.0.1 \
                    --master_port {port} \
                    --nproc_per_node={num_devices} \
                    --use_env \
                    main_dinov2.py \
                    --dataset_file ag_single_dinov2 \
                    --ag_path data/action-genome \
                    --dec_layers_hopd 6 \
                    --dec_layers_interaction 6 \
                    --eval \
                    --pretrained {ckpt_path} \
                    --dsgg_task sgdet \
                    --num_workers 2 \
                    --batch_size 128 \
                    --num_queries 100 \
                    --backbone vit_base_patch14_dinov2 \
                    ")


if __name__ == "__main__":
    device = '0'
    run_eval(device=device, ckpt_path='exps/spatial_sgdet_dinov2_vit_B14/checkpoint_29.pth')
    
