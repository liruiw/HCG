#!/bin/bash

KMP_SETTING="KMP_AFFINITY=granularity=fine,compact,1,0"
KMP_BLOCKTIME=1

export OMP_NUM_THREADS=1
export $KMP_SETTING
export KMP_BLOCKTIME=$KMP_BLOCKTIME

set -x
set -e

DIR="$( cd "$( dirname -- "$0" )" && pwd )"
export PYTHONPATH="${DIR}/../":"$PYTHONPATH"
export PYTHONUNBUFFERED="True"

MODEL_NAME=${1-"03_07_2020_16:10:32"}
RUN_NUM=${2-2}
EPI_NUM=${3-300}
EPOCH=${4-latest}
LOG="output/${MODEL_NAME}/test_log.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

time python -m core.test_offline --pretrained output/${MODEL_NAME} --test   --record  --log \
			 --load_test_scene   --egl    --load_goal --test_script_name $(basename $BASH_SOURCE) \
			 --test_episode_num ${EPI_NUM} --num_runs ${RUN_NUM} --model_surfix ${EPOCH} ;
