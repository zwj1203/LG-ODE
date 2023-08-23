#!/bin/bash
device=6

epoch=300
batch_size=256

lr='5e-4'
train_observe_ratio=0.5
reverse_gt_lambda=0.5
reverse_f_lambda=0.5
test_cut='500'

# LP args
train_cut=('2000' '1600' '1200' '800' '500')
test_observe_ratio=(0.5)


for t in "${train_cut[@]}"
  do
      for test_or in "${test_observe_ratio[@]}"
      do
          python run_models.py --lr $lr --reverse_gt_lambda $reverse_gt_lambda --reverse_f_lambda $reverse_f_lambda --niters $epoch --train_cut $t --test_cut $test_cut --sample-percent-train $train_observe_ratio --sample-percent-test $test_or --device $device --batch-size $batch_size
      done
  done
