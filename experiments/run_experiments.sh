#!/bin/bash -x

# Runs experiments needed for Figure 5.
# Plots are generated with analysis.py.
# Note that for the paper, we ran these experiments with a slightly different script on a cluster:
#     This script is likely to be impractical to run on a single machine.

trials=25
    
for dataset in `python -c 'import util; print(" ".join(util.atk_dataset_names() + [x for x in util.pmlb_binary_classification_dataset_names() if x not in util.atk_dataset_names()]))'` \
               `python -c 'import util; print(" ".join(util.openml_binary_classification_dataset_names()))'`;
do

    echo $dataset

    for degree in 1
    do
        for model in linear dense
        do

            mkdir -p log/$model$degree/$dataset

            for loss in logistic hinge
            do
                ./main.py $dataset --restart --log info --trials $trials --$loss --$model --degree $degree --mode average     --epochs 10000 --save --logfile log/$model$degree/${dataset}/${loss}_average
                ./main.py $dataset --restart --log info --trials $trials --$loss --$model --degree $degree --mode atk         --epochs 10000 --save --logfile log/$model$degree/${dataset}/${loss}_atk
                ./main.py $dataset --restart --log info --trials $trials --$loss --$model --degree $degree --mode top         --epochs 10000 --save --logfile log/$model$degree/${dataset}/${loss}_top
                ./main.py $dataset --restart --log info --trials $trials --$loss --$model --degree $degree --mode close       --epochs 10000 --save --logfile log/$model$degree/${dataset}/${loss}_close
                ./main.py $dataset --restart --log info --trials $trials --$loss --$model --degree $degree --mode close_decay --epochs 10000 --save --logfile log/$model$degree/${dataset}/${loss}_close_decay --tol 0
            done
        done
    done
done
