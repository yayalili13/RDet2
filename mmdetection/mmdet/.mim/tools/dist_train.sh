#!/usr/bin/env bash

CONFIG=$1
GPUS=$2
CHECKPOINT=$3
PORT=${PORT:-29500}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:3}

# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
# python -m torch.distributed.run --nproc_per_node=$GPUS --master_port=$PORT \
#    $(dirname "$0")/train.py $CONFIG --auto-resume --launcher pytorch ${@:3}

# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
# python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
#    $(dirname "$0")/train.py $CONFIG --resume-from $CHECKPOINT --launcher pytorch ${@:4}
