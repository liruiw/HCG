#!/bin/bash
set -x
set -e

DIR="$( cd "$( dirname -- "$0" )" && pwd )"
export PYTHONUNBUFFERED="True"
LOG_NAME="agent"
LOG="output_misc/logs/$LOG_NAME.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"
SCRIPT_NAME=${1-"ddpg_finetune.yaml"}
POLICY_NAME=${2-"BC"}
PRETRAINED=${3-"" } #
MODEL_NAME=${4-"`date +'%d_%m_%Y_%H:%M:%S'`" }
SURFIX=${5-"latest"}

time python -m core.train_online  --save_model  \
		  --policy ${POLICY_NAME} --log  --fix_output_time ${MODEL_NAME}  \
		   --finetune   --pretrained output/${PRETRAINED} \
		  --config_file ${SCRIPT_NAME} --load_buffer --load_online_buffer   --visdom

bash ./experiments/scripts/test_ycb_gen_12_mpc.sh  ${MODEL_NAME}