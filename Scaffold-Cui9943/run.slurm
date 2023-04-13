#!/bin/bash
#SBATCH --job-name=scaffoldCui
#SBATCH --partition=cu-1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --error=%j.err
#SBATCH --output=%j.out
##########################################
I=`whoami`
CURDIR=`pwd`
rm -rf $CURDIR/nodelist.$SLURM_JOB_ID
NODES=`scontrol show hostnames $SLURM_JOB_NODELIST`
for i in $NODES
do
echo "$i:$SLURM_NTASKS_PER_NODE" >> $CURDIR/nodelist.$SLURM_JOB_ID
done
echo $SLURM_NPROCS
echo "process will start at : "
date
echo "++++++++++++++++++++++++++++++++++++++++"

export PATH="/lustre/home/qfchen/anaconda3/bin:$PATH"
export PATH="/lustre/home/qfchen/anaconda3/envs/RDKit/:$PATH"
export PATH="/lustre/home/qfchen/anaconda3/envs/RDKit/bin/:$PATH"


python main.py

echo "++++++++++++++++++++++++++++++++++++++++"
echo "process end at : "
date
sleep 300
