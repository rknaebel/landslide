#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=$1
DATA=$2
AREA=$3
TEST=$4
BS=$5

python3 main.py --modes eval --data $DATA \
                --model cifar_as${AREA}_p${P//./}_t${TEST} \
                --tmp \
                --area $AREA --batch $BS \
                --test ${TEST}

python3 main.py --modes eval --data $DATA \
                --model maxout_as${AREA}_p${P//./}_t${TEST} \
                --tmp \
                --area $AREA --batch $BS \
                --test ${TEST}

python3 main.py --modes eval --data $DATA \
                --model inception_as${AREA}_p${P//./}_t${TEST} \
                --tmp \
                --area $AREA --batch $BS \
                --test ${TEST}

python3 main.py --modes eval --data $DATA \
                --model resnet_as${AREA}_p${P//./}_t${TEST} \
                --tmp \
                --area $AREA --batch $BS \
                --test ${TEST}
