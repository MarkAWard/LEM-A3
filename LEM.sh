#!/bin/bash

#PBS -l nodes=1:ppn=1:gpus=1:titan
#PBS -l walltime=12:00:00
#PBS -l mem=100GB
#PBS -N a3
#PBS -M es3431@nyu.edu


mkdir $HOME/LEM
cp /scratch/es3431/A3/* $HOME/LEM

