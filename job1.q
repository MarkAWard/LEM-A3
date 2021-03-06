#!/bin/bash
#PBS -l nodes=1:ppn=1:gpus=1:titan
#PBS -l walltime=12:00:00
#PBS -l mem=100GB
#PBS -N baseline
#PBS -M ez466@nyu.edu
#PBS -m bea


module purge
module load cuda/6.5.12

cd $SCRATCH/DL/A3

th 0_K80_options.lua -mode train -gpudevice 1 -bufferPath_x data/TR_x_100x200.t7b -bufferPath_y data/TR_y_100x200.t7b >>job.log


