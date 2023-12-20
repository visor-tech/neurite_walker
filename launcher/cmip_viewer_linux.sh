#!/bin/bash

# load python env
source /media/xyy/DATA/pyenvs/visor/bin/activate

read -p "Please enter a numerical ID of the neuron: " userinput

echo "Starting Lychnis"
export LychnisServerPort=29738

script_path=$(dirname "$(realpath "$0")")

echo "Starting CMIP viewer"
export MPLBACKEND=qtagg
export PYTHONPATH="$script_path/code/SimpleVolumeViewer/"

python \
$script_path/code/neurite_walker/neu_walk.py \
--zarr_dir "$script_path/dataset/RM009/blk128_neu210_sps231209.zarr" \
--cmip_dir "$script_path/dataset/RM009/cmip_swc210_res1" \
--res 1 \
--view_length 2000/3 \
--filter "(branch_depth(processes)<=7) & (path_length_to_root(end_point(processes))>10000)" \
--viewer neu3dviewer \
--view "$script_path/dataset/RM009/swc210_largest/neuron#$userinput.lyp.swc"
