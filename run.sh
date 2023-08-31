#!/bin/bash
device=4

epoch=2000
batch_size=512

lr='1e-4'
train_observe_ratio=0.6
reverse_gt_lambda=0
reverse_f_lambda=0.5
test_cut='400'
data='charged'  #help="spring,charged,motion"
Tmax=2000
eta_min=1e-6



# LP args
train_cut=('2000' '1500' '1000' '750' '600' '300' '100'  )
test_observe_ratio=0.6


for t in "${train_cut[@]}"
  do
      for test_or in "${test_observe_ratio[@]}"
      do
          python run_models.py --data $data --Tmax $Tmax --eta_min $eta_min --lr $lr --reverse_gt_lambda $reverse_gt_lambda --reverse_f_lambda $reverse_f_lambda --niters $epoch --train_cut $t --test_cut $test_cut --sample-percent-train $train_observe_ratio --sample-percent-test $test_or --device $device --batch-size $batch_size
      done
  done
