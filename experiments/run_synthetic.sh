#!/bin/bash

# Generates Figure 1.

./main.py easy  --mode close average --labels close,atk,maximal average --reg 0 --split false --normalize false --epochs 10000 --render true --lr 5e-6 --synthetic true
./main.py imbalance  --mode average close --reg 0 --split false --normalize false --epochs 10000 --render true --lr 5e-6 --synthetic true
./main.py imbalance+outlier  --mode average maximal close --reg 0 --split false --normalize false --epochs 10000 --render true --lr 5e-6 --synthetic true
./main.py overlap  --mode average maximal atk close --reg 0 --split false --normalize false --epochs 10000 --render true --lr 1e-5 --synthetic true
