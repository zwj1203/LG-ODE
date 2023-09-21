#!/bin/bash
device=0

batch_size=512
n_balls=5
data='simple_spring'
train_cut="20000"
test_cut='5000'


epoch=150
lr='1e-4'
Tmax=200
eta_min=1e-9

Observe_ratio=('0.4' '0.6' '0.8' )

reverse_gt_lambda=0
reverse_f_lambda=0.5


for observe_ratio in "${Observe_ratio[@]}"
  do
    python run_models.py --data $data --n-balls $n_balls --Tmax $Tmax --eta_min $eta_min --lr $lr  --reverse_gt_lambda $reverse_gt_lambda --reverse_f_lambda $reverse_f_lambda --niters $epoch --train_cut $train_cut --test_cut $test_cut --sample-percent-train $observe_ratio --sample-percent-test $observe_ratio --device $device --batch-size $batch_size
  done