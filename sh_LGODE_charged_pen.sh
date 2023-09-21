#!/bin/bash
device=4

batch_size=512


train_cut="20000"
test_cut='5000'
reverse_gt_lambda=0
reverse_f_lambda=0

epoch=300
lr='1e-5'
Tmax=300
eta_min=1e-9

Observe_ratio=('0.4' '0.6' '0.8' )
Data=('charged' 'pendulum')
N_balls=('5' '3')

length=${#Data[@]}

for ((i = 0; i < length; i++)); do
  data="${Data[i]}"
  n_balls="${N_balls[i]}"
  for observe_ratio in "${Observe_ratio[@]}"
    do
      python run_models.py --data $data --n-balls $n_balls --Tmax $Tmax --eta_min $eta_min --lr $lr  --reverse_gt_lambda $reverse_gt_lambda --reverse_f_lambda $reverse_f_lambda --niters $epoch --train_cut $train_cut --test_cut $test_cut --sample-percent-train $observe_ratio --sample-percent-test $observe_ratio --device $device --batch-size $batch_size
    done
  done


