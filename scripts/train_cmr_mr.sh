#!/bin/bash
# train a model to segment cardiac MRI (MSCMR-bSSFP)
GPUID1=0
export CUDA_VISIBLE_DEVICES=$GPUID1

###### Shared configs ######
DATASET='CMR'
NWORKER=8
RUNS=1
ALL_EV=(0 1 2 3 4) # 2-fold cross validation (0, 1, 2, 3, 4)
TEST_LABEL=[1.2,3]
EXCLUDE_LABEL=None
USE_GT=False
###### Training configs ######
NSTEP=30000
DECAY=0.98

MAX_ITER=1000 # defines the size of an epoch
SNAPSHOT_INTERVAL=10000 # interval for saving snapshot
SEED=2021

echo ========================================================================

for EVAL_FOLD in "${ALL_EV[@]}"
do
  PREFIX="train_${DATASET}_cv${EVAL_FOLD}"
  echo $PREFIX
  LOGDIR="./exps_on_${DATASET}_FSS"

  if [ ! -d $LOGDIR ]
  then
    mkdir -p $LOGDIR
  fi

  python3 train.py with \
  mode='train' \
  dataset=$DATASET \
  num_workers=$NWORKER \
  n_steps=$NSTEP \
  eval_fold=$EVAL_FOLD \
  test_label=$TEST_LABEL \
  exclude_label=$EXCLUDE_LABEL \
  use_gt=$USE_GT \
  max_iters_per_load=$MAX_ITER \
  seed=$SEED \
  save_snapshot_every=$SNAPSHOT_INTERVAL \
  lr_step_gamma=$DECAY \
  path.log_dir=$LOGDIR
done
