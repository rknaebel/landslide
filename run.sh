#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=$1
DATA=/home/ahmed/data/
P=$2
AREA=$3
TEST=$4

python3 main.py --mode train --data $DATA \
                --type cifar --model cifar_as${AREA}_p${P//./}_t${TEST} \
                --h5data /tmp/rk/data${TEST}.h5 --tmp \
                --area $AREA --samples 1000000 --batch 4096 \
                --p $P --epochs 100 --test ${TEST}

python3 main.py --mode train --data $DATA \
                --type medium_maxout --model maxout_as${AREA}_p${P//./}_t${TEST} \
                --h5data /tmp/rk/data${TEST}.h5 --tmp \
                --area $AREA --samples 1000000 --batch 4096 \
                --p $P --epochs 100 --test ${TEST}

python3 main.py --mode train --data $DATA \
                --type inception --model inception_as${AREA}_p${P//./}_t${TEST} \
                --h5data /tmp/rk/data${TEST}.h5 --tmp \
                --area $AREA --samples 1000000 --batch 4096 \
                --p $P --epochs 100 --test ${TEST}

python3 main.py --mode train --data $DATA \
                --type resnet --model resnet_as${AREA}_p${P//./}_t${TEST} \
                --h5data /tmp/rk/data${TEST}.h5 --tmp \
                --area $AREA --samples 1000000 --batch 4096 \
                --p $P --epochs 100 --test ${TEST}
