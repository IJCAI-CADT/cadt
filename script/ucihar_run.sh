#!/bin/bash
python main.py \
  --x_dim  9\
  --n_class 6\
  --seq_len 128\
  --n_epoch 351\
  --batch_size 128\
  --device 0\
  --ball_loss_weight 10\
  --discriminator_loss 1\
  --source_domain 2\
  --target_domain 4\
  --dataset 'ucihar'