#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=$1
DATA=$2
P=$3
AREA=$4
TEST=$5

BS=2048

python3 main.py --modes train --data $DATA \
                --type cifar --model cifar_as${AREA}_p${P//./}_t${TEST} \
                --h5data /tmp/rk/data${TEST}.h5 --tmp \
                --area $AREA --samples 1000000 --batch $BS \
                --p $P --p_val 0.001 --epochs 100 --test ${TEST}

python3 main.py --modes train --data $DATA \
                --type maxout --model maxout_as${AREA}_p${P//./}_t${TEST} \
                --h5data /tmp/rk/data${TEST}.h5 --tmp \
                --area $AREA --samples 1000000 --batch $BS \
                --p $P --p_val 0.001 --epochs 100 --test ${TEST}

python3 main.py --modes train --data $DATA \
                --type inception --model inception_as${AREA}_p${P//./}_t${TEST} \
                --h5data /tmp/rk/data${TEST}.h5 --tmp \
                --area $AREA --samples 1000000 --batch $BS \
                --p $P --p_val 0.001 --epochs 100 --test ${TEST}

python3 main.py --modes train --data $DATA \
                --type resnet --model resnet_as${AREA}_p${P//./}_t${TEST} \
                --h5data /tmp/rk/data${TEST}.h5 --tmp \
                --area $AREA --samples 1000000 --batch $BS \
                --p $P --p_val 0.001 --epochs 100 --test ${TEST}
