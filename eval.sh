#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=$1
DATA=$2
P=$3
AREA=$4
TEST=$5
BS=$6

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
