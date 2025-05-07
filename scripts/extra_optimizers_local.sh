#!/bin/sh


python3 src/train_imageclassification.py --dataset=MNIST --model=squeezenet1_0 --epochs=100 --batch_size=64 --lr=0.001 --optimizer=adabelief
python3 src/train_imageclassification.py --dataset=MNIST --model=squeezenet1_0 --epochs=100 --batch_size=64 --lr=0.001 --optimizer=adamp
python3 src/train_imageclassification.py --dataset=MNIST --model=squeezenet1_0 --epochs=100 --batch_size=64 --lr=0.001 --optimizer=madgrad
python3 src/train_imageclassification.py --dataset=MNIST --model=squeezenet1_0 --epochs=100 --batch_size=64 --lr=0.001 --optimizer=adan
python3 src/train_imageclassification.py --dataset=MNIST --model=squeezenet1_0 --epochs=100 --batch_size=64 --lr=0.001 --optimizer=lion
