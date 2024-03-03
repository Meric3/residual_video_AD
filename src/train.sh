#!/bin/bash

# CUDA_VISIBLE_DEVICES=1,2,3 python train_generator.py --dataset exit --input_channel 4 --image_size 224 --timesteps 3 --multi_gpu --model convLSTM_res_attention_pred --ngpus 2 --batch_size 8 --epochs 100 --checkname convLSTM_res_flow 


# ped 1 
# CUDA_VISIBLE_DEVICES=0,2 python train_generator.py --dataset ped1 --input_channel 4 --image_size 224 --timesteps 6 --multi_gpu --model convLSTM_res_attention_pred --ngpus 2 --batch_size 8 --epochs 100 --checkname convLSTM_res_flow 


# ped1 timestpep3
# CUDA_VISIBLE_DEVICES=0,3 python train_generator.py --dataset ped1 --input_channel 4 --image_size 224 --timesteps 3 --multi_gpu --model convLSTM_res_attention_pred --ngpus 2 --batch_size 4 --epochs 100 --checkname convLSTM_res_flow 

# ped 1 timestep3 batch 16
# CUDA_VISIBLE_DEVICES=0,3 python train_generator.py --dataset ped1 --input_channel 4 --image_size 224 --timesteps 3 --multi_gpu --model convLSTM_res_attention_pred --ngpus 2 --batch_size 16 --epochs 100 --checkname convLSTM_res_flow 


# exit step4 
# CUDA_VISIBLE_DEVICES=0,3 python train_generator.py --dataset exit --input_channel 4 --image_size 224 --timesteps 4 --multi_gpu --model convLSTM_res_attention_pred --ngpus 2 --batch_size 8 --epochs 100 --checkname convLSTM_res_flow 

# exit step5
# CUDA_VISIBLE_DEVICES=0,3 python train_generator.py --dataset exit --input_channel 4 --image_size 224 --timesteps 5 --multi_gpu --model convLSTM_res_attention_pred --ngpus 2 --batch_size 8 --epochs 100 --checkname convLSTM_res_flow 


# shang  
CUDA_VISIBLE_DEVICES=0,1 python train_generator.py --dataset shang --input_channel 4 --image_size 224 --timesteps 4 --multi_gpu --model convLSTM_res_attention_pred --ngpus 2 --batch_size 8 --epochs 100 --checkname convLSTM_res_flow 

CUDA_VISIBLE_DEVICES=2,3 python train_generator.py --dataset shang --input_channel 4 --image_size 224 --timesteps 3 --multi_gpu --model convLSTM_res_attention_pred --ngpus 2 --batch_size 8 --epochs 100 --checkname convLSTM_res_flow 
