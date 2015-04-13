#!/bin/bash

#PBS -l nodes=1:ppn=1:gpus=1:titan
#PBS -l walltime=12:00:00
#PBS -l mem=100GB
#PBS -N a3
#PBS -M es3431@nyu.edu

module purge
module load cuda/6.5.12
module load torch-deps/7

cd $HOME

th 0_K80_options.lua -mode train -gpudevice 1 -bufferPath_x data/TR_x_100x200.t7b -bufferPath_y data/TR_y_100x200.t7b >>job.log



th 0_K80_options.lua -mode train -bufferPath_x data/TR_x_100x200.t7b -bufferPath_y data/TR_y_100x200.t7b -learningRate 0.05 -momentum 0.5 -learningRateDecay 0.001 -seed 11

