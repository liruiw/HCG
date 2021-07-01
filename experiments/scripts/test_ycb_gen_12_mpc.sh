#!/bin/bash

KMP_SETTING="KMP_AFFINITY=granularity=fine,compact,1,0"
set -x
set -e

DIR="$( cd "$( dirname -- "$0" )" && pwd )"
export PYTHONUNBUFFERED="True"
LOG_NAME="agent"

MODEL_NAME=${1-"03_07_2020_16:10:32"}
RUN_NUM=${2-2}
EPI_NUM=${3-300}
EPOCH=${4-latest}
LOG="outputs/${MODEL_NAME}/test_log.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

time python -m core.test_offline --pretrained output/${MODEL_NAME} --test  --record  --log \
			 --load_test_scene   --egl    --load_goal --use_sample_latent --sample_latent_gap 12 --test_script_name $(basename $BASH_SOURCE) \
			 --test_episode_num ${EPI_NUM} --num_runs ${RUN_NUM} --model_surfix ${EPOCH} --critic_mpc --multi_traj_sample;
