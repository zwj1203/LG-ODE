#!/bin/bash
device=4
batch_size=512
n_balls=5

epoch=200


#lr
lr='1e-4'
Tmax=200
eta_min=1e-9


#observe ratio
Observe_ratio=('0.4' '0.6' '0.8' )
#'0.4' '0.6' '0.8'


#reverse_lambda
reverse_gt_lambda=0
reverse_f_lambda=0.1
energy_lambda=0

data='damped_spring'  #help="simple_spring,damped_spring,forced_spring,charged,pendulum"
test_cut='5000'

# LP args
train_cut="20000"

[]
#for t in "${train_cut[@]}"
#  do
#      for test_or in "${test_observe_ratio[@]}"
#      do
#          python run_models.py --data $data --Tmax $Tmax --eta_min $eta_min --lr $lr --energy_lambda $energy_lambda --reverse_gt_lambda $reverse_gt_lambda --reverse_f_lambda $reverse_f_lambda --niters $epoch --train_cut $train_cut --test_cut $test_cut --sample-percent-train $observe_ratio --sample-percent-test $observe_ratio --device $device --batch-size $batch_size
#      done
#  done
for observe_ratio in "${Observe_ratio[@]}"
  do
    python run_models.py --data $data --n-balls $n_balls --Tmax $Tmax --eta_min $eta_min --lr $lr --energy_lambda $energy_lambda --reverse_gt_lambda $reverse_gt_lambda --reverse_f_lambda $reverse_f_lambda --niters $epoch --train_cut $train_cut --test_cut $test_cut --sample-percent-train $observe_ratio --sample-percent-test $observe_ratio --device $device --batch-size $batch_size
  done