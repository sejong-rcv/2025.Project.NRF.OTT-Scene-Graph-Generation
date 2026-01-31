import os
import sys
import random

def run_eval(device='0', ckpt_path='exps/spatial_sgdet_dinov2/checkpoint_39.pth'):
    port = random.randint(10000,59999)
    num_devices = len(device.split(','))
    os.environ['CUDA_VISIBLE_DEVICES'] = device
    os.system(f"python -m torch.distributed.launch \
                    --master_addr 127.0.0.1 \
                    --master_port {port} \
                    --nproc_per_node={num_devices} \
                    --use_env \
                    main_dinov2.py \
                    --dataset_file ag_single \
                    --ag_path data/action-genome \
                    --backbone resnet50 \
                    --num_queries 100 \
                    --dec_layers_hopd 6 \
                    --dec_layers_interaction 6 \
                    --batch_size 32 \
                    --eval \
                    --pretrained {ckpt_path} \
                    --num_workers 8\
                    --dsgg_task sgdet")


if __name__ == "__main__":
    device = '3'
    run_eval(device=device, ckpt_path='exps/spatial_sgdet_dinov2/checkpoint_39.pth')
