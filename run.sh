# evaluation
python3 main.py --mode train --data /home/ahmed/data/ --type simple_conv --model models/model_x1.h5 --h5data /tmp/data.h5 \
                --area 25 \
                --samples 250000 --batch 2048 --p 0.4 --epochs 100

python3 main.py --mode train --data /home/ahmed/data/ --type medium_maxout_conv --model models/model_x2.h5 --h5data /tmp/data.h5 \
                --area 25 \
                --samples 250000 --batch 2048 --p 0.4 --epochs 100

python3 main.py --mode train --data /home/ahmed/data/ --type inception_net --model models/model_x3.h5 --h5data /tmp/data.h5 \
                --area 25 \
                --samples 250000 --batch 2048 --p 0.4 --epochs 100

python3 main.py --mode train --data /home/ahmed/data/ --type resnet --model models/model_x4.h5 --h5data /tmp/data.h5 \
                --area 25 \
                --samples 250000 --batch 2048 --p 0.4 --epochs 100

python3 main.py --mode train --data /home/ahmed/data/ --type cifar --model models/model_x5.h5 --h5data /tmp/data.h5 \
                --area 25 \
                --samples 250000 --batch 2048 --p 0.4 --epochs 100
