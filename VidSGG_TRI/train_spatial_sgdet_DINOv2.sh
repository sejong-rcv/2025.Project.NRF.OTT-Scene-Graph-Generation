#!/bin/bash
export NCCL_IB_DISABLE=1    
export NCCL_P2P_DISABLE=1  
export NCCL_SOCKET_IFNAME=eth0  
export NCCL_DEBUG=WARN
export NCCL_DEBUG_SUBSYS=ALL   

TAG='spatial_sgdet_vit_batches'
PORT=12347

export MASTER_ADDR=127.0.0.1
export MASTER_PORT=${PORT}

NUM_GPUS=4
GPU_IDS=0,1,2,3

export OMP_NUM_THREADS=2
export CUDA_VISIBLE_DEVICES=${GPU_IDS}


# from dsgg scratch
torchrun --nproc_per_node=${NUM_GPUS} \
  --master_addr="127.0.0.1" \
  --master_port="${PORT}" \
  main_dinov2.py \
  --output_dir "exps/${TAG}/" \
  --dataset_file ag_single_dinov3 \
  --ag_path data/action-genome \
  --hidden_dim 256 \
  --dec_layers_hopd 6 \
  --dec_layers_interaction 6 \
  --is_train train \
  --epochs 40 \
  --sched step \
  --scale_lr \
  --lr_backbone 0 \
  --lr 1e-4 \
  --lr_drop 24 32 \
  --warmup_ratio 0.05 \
  --step_gamma 0.1 \
  --min_lr_ratio 0.01 \
  --lr_base_batch 64 \
  --num_workers 2 \
  --batch_size 64 \
  --num_queries 100 \
  --dsgg_task sgdet \
  --backbone vit_small_patch16_dinov3.lvd1689m \
  --pretrained exps/params/sgdet/spatial/checkpoint_22_origin.pth \
  