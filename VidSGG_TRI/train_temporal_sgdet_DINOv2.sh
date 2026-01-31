#!/bin/bash

export NCCL_IB_DISABLE=1              
export NCCL_SOCKET_IFNAME=eth0        
export NCCL_DEBUG=WARN
export NCCL_DEBUG_SUBSYS=ALL          
export TORCH_SHOW_CPP_STACKTRACES=1
export NCCL_ASYNC_ERROR_HANDLING=1

export NCCL_P2P_DISABLE=1       


TAG='temporal_sgdet_vit_batches'
PORT=12352

export MASTER_ADDR=127.0.0.1
export MASTER_PORT=${PORT}

NUM_GPUS=4
GPU_IDS=0,1,2,3

export CUDA_VISIBLE_DEVICES=${GPU_IDS} 
export OMP_NUM_THREADS=2

# from scratch
torchrun --nproc_per_node=${NUM_GPUS} \
  --master_addr="127.0.0.1" \
  --master_port="${PORT}" \
  main_dinov2.py \
  --output_dir "exps/${TAG}/" \
  --dataset_file ag_multi_dinov2 \
  --ag_path data/action-genome \
  --dsgg_task sgdet \
  --freeze_mode 1 \
  --is_train train \
  --dec_layers_hopd 6 \
  --dec_layers_interaction 6 \
  --num_ref_frames 4 \
  --batch_size 32 \
  --epochs 12 \
  --sched step \
  --scale_lr \
  --lr_backbone 0 \
  --lr 1e-4 \
  --lr_drop 6 \
  --warmup_ratio 0.03 \
  --step_gamma 0.1 \
  --lr_base_batch 32 \
  --num_workers 4 \
  --query_temporal_interaction \
  --num_queries 100 \
  --backbone vit_base_patch14_dinov2 \
  --pretrained exps/spatial_sgdet_dinov2_vit_B14/checkpoint_28.pth

