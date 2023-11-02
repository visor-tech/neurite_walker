#!/bin/bash
#Array type job: sbatch --array=1-554:1%72 run_arr1.sh
#SBATCH --job-name=walk_neu
#SBATCH --output=job_%A_%a.out
#SBATCH --error=job_%A_%a.err
#
#SBATCH --partition=compute
#SBATCH --cpus-per-task=1
#SBATCH --time=40:59:59
#SBATCH --mem=3G

#tips
# scancel 18130
# cd ~/dataset/RM009/
# rm -r cmip_swc172_res1
# mkdir cmip_swc172_res1
# cd ~/run
# sbatch --array=1-554:1%36 run_arr1.sh
# https://slurm.schedmd.com/job_array.html
#

# command tips
# sacct -j 18156 | wc -l
# sacct -j 18156 | grep PEND | wc -l
# sacct -j 18156 | grep RUNN | tee | wc -l
# sacct -j 18156 | grep COMP | wc -l

eval "$(/share/home/xiaoyy/mconda3/bin/conda shell.bash hook)"
conda activate py311

hostname
ncpu=`nproc`
npara=1
echo N_CPU = $ncpu , N_PARALLEL = $npara
echo $CONDA_PREFIX
python --version

# before setting OMP_NUM_THREADS=2
# %Cpu(s): 58.1 us, 12.3 sy,  0.0 ni, 28.6 id,  0.0 wa,  0.8 hi,  0.2 si,  0.0 st

echo SLURM_ARRAY_TASK_ID $SLURM_ARRAY_TASK_ID

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# test run
#time find ~/dataset/RM009/swc172_largest -type f -print0 \
#| tail -z -n "+$SLURM_ARRAY_TASK_ID" \
#| head -z -n $npara \
#| xargs -0 -P $npara -n 1 echo 

time find ~/dataset/RM009/swc1.7.2_largest -type f -print0 \
| tail -z -n "+$SLURM_ARRAY_TASK_ID" \
| head -z -n $npara \
| xargs -0 -P $npara -n 1 ~/code/neurite_walker/neu_walk.py --zarr_dir ~/dataset/RM009/blk128_neu172_sps231010.zarr --res 1 --cmip_dir ~/dataset/RM009/cmip_swc172_res1


## run time

# for --cpus-per-task=2
# sacct -j 18156 -o "End, Submit, jobid" | sort
# 2023-10-29T00:07:31 ~ 2023-10-31T01:37:59 = 49.508 h
# n files = 212216

# for --cpus-per-task=1
# scontrol show job 18741
# SubmitTime=2023-11-01T23:39:48
# 

