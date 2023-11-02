#!/bin/bash
# cd ~/run
# sbatch run_s.sh
#
#SBATCH --job-name=walk_neurite
#SBATCH --output=job_%J.out
#SBATCH --error=job_%J.err
#
#SBATCH --partition=compute
#SBATCH --cpus-per-task=72
#SBATCH --time=6-23:59:59
#SBATCH --mem=144G

eval "$(/share/home/xiaoyy/mconda3/bin/conda shell.bash hook)"
conda activate py311

hostname
ncpu=`nproc`
echo NCPU = $ncpu
echo $CONDA_PREFIX
python --version

# test run
#echo -n ~/dataset/RM009/swc172_largest/neuron#1.lyp.swc | xargs -0 -P $ncpu -n 1 ~/code/neurite_walker/neu_walk.py --zarr_dir ~/dataset/RM009/blk128_neu172_sps231010.zarr --res 1 --cmip_dir ~/dataset/RM009/cmip_swc172_res1

time find ~/dataset/RM009/swc172_largest -type f -print0 | xargs -0 -P $ncpu -n 1 ~/code/neurite_walker/neu_walk.py --zarr_dir ~/dataset/RM009/blk128_neu172_sps231010.zarr --res 1 --cmip_dir ~/dataset/RM009/cmip_swc172_res1

