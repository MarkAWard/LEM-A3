#!/bin/bash
#PBS -l nodes=1:ppn=1:gpus=1:titan
#PBS -l walltime=12:00:00
#PBS -l mem=100GB
#PBS -N sgd
#PBS -M maw627@nyu.edu


module purge
module load cuda/6.5.12
module load torch-deps/7

cd $SCRATCH/DL/A3

th 0_K80_options.lua -mode train -bufferPath_x data/TR_x_100x200.t7b -bufferPath_y data/TR_y_100x200.t7b -learningRate 0.05 -momentum 0.5 -learningRateDecay 0.001 -seed 11

