# Adversarial Examples Are a Natural Consequence of Test Error in Noise

This directory contains code for generating the graphs from the paper relating
the performance of a model on Gaussian noise to the distance to the nearest
error.

The CIFAR version of this pipeline is meant to be used with the models trained
by Madry et al. for their paper "Towards Deep Learning Models Resistant to
Adversarial Attacks". Those models can be found by running the code available at
https://github.com/MadryLab/cifar10_challenge/blob/master/fetch_model.py.

Example usage:

~~~~
python -m adv_corr_robust.plots --batch_size=1 --num_batches=1 --model=cifar \
  --model_file=/path/to/madry_cifar_model/checkpoint-70000 \
  --save_dir=/path/to/save_dir --data_dir=/path/to/cifar_data
~~~~
