#!/bin/bash +x

# Runs experiments needed for Figure 5.
# Plots are generated with analysis.py.

dataset=spambase

mkdir -p simulation/${dataset}/outliers
mkdir -p simulation/${dataset}/dup0
mkdir -p simulation/${dataset}/dup1
mkdir -p simulation/${dataset}/hard0
mkdir -p simulation/${dataset}/hard1

for outliers in 0 1 2 8 16 32 64 128 256
do
    echo $outliers
    ./main.py ${dataset} --mode average     --outliers $outliers --trials 25 --pretrain false --log info --logfile simulation/${dataset}/outliers/average_$outliers
    ./main.py ${dataset} --mode atk         --outliers $outliers --trials 25 --pretrain false --log info --logfile simulation/${dataset}/outliers/atk_$outliers
    ./main.py ${dataset} --mode top         --outliers $outliers --trials 25 --pretrain false --log info --logfile simulation/${dataset}/outliers/top_$outliers
    ./main.py ${dataset} --mode maximal     --outliers $outliers --trials 25 --pretrain false --log info --logfile simulation/${dataset}/outliers/maximal_$outliers
    ./main.py ${dataset} --mode close       --outliers $outliers --trials 25 --pretrain false --log info --logfile simulation/${dataset}/outliers/close_$outliers
    ./main.py ${dataset} --mode close_decay --outliers $outliers --trials 25 --pretrain false --log info --logfile simulation/${dataset}/outliers/close_decay_$outliers
done

for dup in 0 100 200 400 800 1600 3200 6400 12800
do
    echo $dup
    ./main.py ${dataset} --mode average     --dup0 $dup --trials 25 --log info --logfile simulation/${dataset}/dup0/average_$dup
    ./main.py ${dataset} --mode atk         --dup0 $dup --trials 25 --log info --logfile simulation/${dataset}/dup0/atk_$dup
    ./main.py ${dataset} --mode top         --dup0 $dup --trials 25 --log info --logfile simulation/${dataset}/dup0/top_$dup
    ./main.py ${dataset} --mode maximal     --dup0 $dup --trials 25 --log info --logfile simulation/${dataset}/dup0/maximal_$dup
    ./main.py ${dataset} --mode close       --dup0 $dup --trials 25 --log info --logfile simulation/${dataset}/dup0/close_$dup
    ./main.py ${dataset} --mode close_decay --dup0 $dup --trials 25 --log info --logfile simulation/${dataset}/dup0/close_decay_$dup

    ./main.py ${dataset} --mode average     --dup1 $dup --trials 25 --log info --logfile simulation/${dataset}/dup1/average_$dup
    ./main.py ${dataset} --mode atk         --dup1 $dup --trials 25 --log info --logfile simulation/${dataset}/dup1/atk_$dup
    ./main.py ${dataset} --mode top         --dup1 $dup --trials 25 --log info --logfile simulation/${dataset}/dup1/top_$dup
    ./main.py ${dataset} --mode maximal     --dup1 $dup --trials 25 --log info --logfile simulation/${dataset}/dup1/maximal_$dup
    ./main.py ${dataset} --mode close       --dup1 $dup --trials 25 --log info --logfile simulation/${dataset}/dup1/close_$dup
    ./main.py ${dataset} --mode close_decay --dup1 $dup --trials 25 --log info --logfile simulation/${dataset}/dup1/close_decay_$dup
done

for hard in 0 100 200 400 800
do
    echo $hard
    ./main.py ${dataset} --mode average     --hard0 $hard --trials 25 --log info --logfile simulation/${dataset}/hard0/average_$hard
    ./main.py ${dataset} --mode atk         --hard0 $hard --trials 25 --log info --logfile simulation/${dataset}/hard0/atk_$hard
    ./main.py ${dataset} --mode top         --hard0 $hard --trials 25 --log info --logfile simulation/${dataset}/hard0/top_$hard
    ./main.py ${dataset} --mode maximal     --hard0 $hard --trials 25 --log info --logfile simulation/${dataset}/hard0/maximal_$hard
    ./main.py ${dataset} --mode close       --hard0 $hard --trials 25 --log info --logfile simulation/${dataset}/hard0/close_$hard
    ./main.py ${dataset} --mode close_decay --hard0 $hard --trials 25 --log info --logfile simulation/${dataset}/hard0/close_decay_$hard

    ./main.py ${dataset} --mode average     --hard1 $hard --trials 25 --log info --logfile simulation/${dataset}/hard1/average_$hard
    ./main.py ${dataset} --mode atk         --hard1 $hard --trials 25 --log info --logfile simulation/${dataset}/hard1/atk_$hard
    ./main.py ${dataset} --mode top         --hard1 $hard --trials 25 --log info --logfile simulation/${dataset}/hard1/top_$hard
    ./main.py ${dataset} --mode maximal     --hard1 $hard --trials 25 --log info --logfile simulation/${dataset}/hard1/maximal_$hard
    ./main.py ${dataset} --mode close       --hard1 $hard --trials 25 --log info --logfile simulation/${dataset}/hard1/close_$hard
    ./main.py ${dataset} --mode close_decay --hard1 $hard --trials 25 --log info --logfile simulation/${dataset}/hard1/close_decay_$hard
done
