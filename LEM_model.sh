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
th submissionScript.lua -model model_elad_run_simple_epoch_9.net
